"""
AIGateway — central orchestrator for all LLM provider calls.

Responsibilities:
  - Route analyst requests to the configured provider
  - Enforce retry logic with exponential backoff
  - Validate every response via ResponseValidator
  - Apply circuit breaker to prevent cascading failures
  - Track metrics (latency, tokens, cache hits, errors)
  - Return validated Pydantic instances or fallback objects

No caller should ever interact with a provider directly — always through the gateway.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from market_intelligence_ai.providers.base import (
    MarketIntelligenceProvider,
    ProviderError,
    ProviderRateLimitError,
)
from market_intelligence_ai.response_validator.validator import (
    ResponseValidator,
    ValidationFailure,
)
from market_intelligence_ai.ai_gateway.circuit_breaker import CircuitBreaker
from market_intelligence_ai.utils.config import mia_config
from market_intelligence_ai.utils.logger import logger
from market_intelligence_ai.utils.metrics import GatewayMetrics

T = TypeVar("T", bound=BaseModel)

_RETRY_BACKOFF_S = [0.5, 2.0, 5.0]   # Delays for attempt 1, 2, 3


class AIGateway:
    """
    Validates, retries, and rate-controls all AI provider calls.

    Each analyst instantiates the gateway (or shares a singleton from the engine).
    """

    def __init__(self, provider: MarketIntelligenceProvider) -> None:
        self._provider        = provider
        self._circuit_breaker = CircuitBreaker(
            threshold=mia_config.MIA_CIRCUIT_BREAKER_THRESHOLD,
            reset_timeout_s=mia_config.MIA_CIRCUIT_RESET_SECONDS,
        )
        self._metrics = GatewayMetrics()
        self._rate_limited_until: Optional[datetime] = None  # skip calls until token quota replenishes

    # ── Public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        system_prompt:  str,
        user_prompt:    str,
        model:          str,
        schema:         Type[T],
        fallback_data:  dict[str, Any],
        max_retries:    int = 2,
        temperature:    float = 0.1,
        max_tokens:     int = 1024,
    ) -> T:
        """
        Execute a provider call, validate the response, and return a typed instance.

        On validation failure, retries with a repair prompt (up to max_retries).
        On provider failure or circuit open, returns a fallback instance.

        Args:
            system_prompt: The system context for the LLM
            user_prompt:   The analyst-specific user prompt
            model:         Model name to use
            schema:        Pydantic model class to validate the response against
            fallback_data: Data for constructing a fallback when all retries fail
            max_retries:   Number of retry attempts on validation failure
            temperature:   LLM temperature (low = more deterministic)
            max_tokens:    Maximum tokens for completion

        Returns:
            Validated instance of `schema`, or a fallback instance with is_fallback=True
        """
        if self._circuit_breaker.is_open:
            logger.warning("AIGateway: circuit OPEN — returning fallback for {}", schema.__name__)
            await self._metrics.record_failure()
            return ResponseValidator.make_fallback(schema, {**fallback_data, "is_fallback": True})

        now = datetime.now(timezone.utc)
        if self._rate_limited_until and now < self._rate_limited_until:
            remaining = (self._rate_limited_until - now).total_seconds()
            logger.warning(
                "AIGateway: token quota window active — skipping call, retry in {:.0f}s",
                remaining,
            )
            await self._metrics.record_failure()
            return ResponseValidator.make_fallback(schema, {**fallback_data, "is_fallback": True})

        last_validation_error: Optional[str] = None
        current_user_prompt = user_prompt

        for attempt in range(max_retries + 1):
            try:
                response = await self._provider.complete(
                    system_prompt=system_prompt,
                    user_prompt=current_user_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                validated = ResponseValidator.validate(response.content, schema)
                # Inject server-side audit fields the AI doesn't produce
                if hasattr(validated, "latency_ms"):
                    validated.latency_ms = response.latency_ms
                self._circuit_breaker.record_success()
                retries = attempt  # attempts before success
                await self._metrics.record_provider_call(
                    latency_ms=response.latency_ms,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    retries=retries,
                )
                if attempt > 0:
                    logger.debug("AIGateway: validated on retry #{}", attempt)
                return validated

            except ValidationFailure as exc:
                last_validation_error = str(exc)
                logger.warning(
                    "AIGateway: validation failure (attempt {}/{}) — {}",
                    attempt + 1, max_retries + 1, str(exc)[:120],
                )
                if attempt < max_retries:
                    # Build repair prompt for next attempt
                    current_user_prompt = ResponseValidator.build_repair_prompt(
                        user_prompt, last_validation_error
                    )
                    await asyncio.sleep(_RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)])
                    continue

            except ProviderRateLimitError as exc:
                retry_after = getattr(exc, "retry_after_s", None)
                # If the provider says to wait longer than our cycle, don't waste retries —
                # just return fallback and let the next scheduled cycle try again.
                if retry_after and retry_after > 60:
                    self._rate_limited_until = datetime.now(timezone.utc) + timedelta(seconds=retry_after)
                    logger.warning(
                        "AIGateway: token quota — retry-after {:.0f}s, calls paused until {}",
                        retry_after,
                        self._rate_limited_until.strftime("%H:%M:%S UTC"),
                    )
                    break  # exit retry loop → return fallback (circuit NOT tripped)
                logger.warning("AIGateway: rate limit (attempt {}) — {}", attempt + 1, exc)
                backoff = retry_after or _RETRY_BACKOFF_S[min(attempt, 2)]
                if attempt < max_retries:
                    await asyncio.sleep(backoff)
                    continue
                # Don't trip circuit breaker on rate limits — it's quota exhaustion, not provider failure

            except ProviderError as exc:
                logger.error("AIGateway: provider error (attempt {}) — {}", attempt + 1, exc)
                self._circuit_breaker.record_failure()
                if attempt < max_retries:
                    await asyncio.sleep(_RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)])
                    continue

        # All retries exhausted
        logger.error("AIGateway: all retries exhausted for {} — returning fallback", schema.__name__)
        await self._metrics.record_failure(retries=max_retries)
        return ResponseValidator.make_fallback(schema, {**fallback_data, "is_fallback": True})

    @property
    def metrics(self) -> GatewayMetrics:
        return self._metrics

    @property
    def circuit_state(self) -> str:
        return self._circuit_breaker.state

    @property
    def provider(self) -> MarketIntelligenceProvider:
        return self._provider
