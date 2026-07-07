"""
GroqProvider — Groq LLM provider implementation.

Uses the official groq Python SDK with AsyncGroq for non-blocking calls.
Supports JSON mode via response_format={"type": "json_object"}.

Model recommendations:
  llama-3.3-70b-versatile — complex analysis (headline, event, narrative)
  llama-3.1-8b-instant    — fast classification (contradiction detector)
"""
from __future__ import annotations

import time
from typing import Optional

from market_intelligence_ai.providers.base import (
    MarketIntelligenceProvider,
    ProviderError,
    ProviderHealth,
    ProviderRateLimitError,
    ProviderAuthError,
    ProviderResponse,
)
from market_intelligence_ai.utils.logger import logger

try:
    from groq import AsyncGroq
    from groq import APIStatusError, RateLimitError, AuthenticationError
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    AsyncGroq = None  # type: ignore[assignment,misc]
    APIStatusError = Exception  # type: ignore[assignment,misc]
    RateLimitError = Exception  # type: ignore[assignment,misc]
    AuthenticationError = Exception  # type: ignore[assignment,misc]


class GroqProvider(MarketIntelligenceProvider):
    """
    Groq-backed market intelligence provider.

    Falls back gracefully when the groq SDK is not installed or no API key
    is configured — callers see a ProviderError with a clear message.
    """

    _PROVIDER_NAME    = "groq"
    _PROVIDER_VERSION = "groq_v1"
    _DEFAULT_MODEL    = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: Optional[object] = None

        if not _GROQ_AVAILABLE:
            logger.warning("groq SDK not installed — GroqProvider will not function. pip install groq")
            return

        if api_key:
            self._client = AsyncGroq(api_key=api_key)
            logger.info("GroqProvider initialized with model={}", self._DEFAULT_MODEL)
        else:
            logger.warning("GROQ_API_KEY not set — GroqProvider operating in degraded mode")

    # ── Interface implementation ───────────────────────────────────────────────

    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        model:         str,
        temperature:   float,
        max_tokens:    int,
    ) -> ProviderResponse:
        if not _GROQ_AVAILABLE:
            raise ProviderError("groq SDK not installed — run: pip install groq>=0.7.0")
        if self._client is None:
            raise ProviderError("GROQ_API_KEY not configured — cannot call Groq API")

        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except RateLimitError as exc:
            # Extract retry_after_s from the response headers if available
            retry_after: float | None = None
            try:
                raw = getattr(exc, "response", None)
                if raw is not None:
                    ra = raw.headers.get("retry-after") or raw.headers.get("x-ratelimit-reset-tokens")
                    if ra:
                        retry_after = float(ra)
            except Exception:
                pass
            raise ProviderRateLimitError(f"Groq rate limit: {exc}", retry_after_s=retry_after) from exc
        except AuthenticationError as exc:
            raise ProviderAuthError(f"Groq auth failure: {exc}") from exc
        except APIStatusError as exc:
            raise ProviderError(f"Groq API error ({exc.status_code}): {exc.message}", exc.status_code) from exc
        except Exception as exc:
            raise ProviderError(f"Groq unexpected error: {exc}") from exc

        elapsed_ms = (time.monotonic() - t0) * 1000

        choice = response.choices[0]
        usage  = response.usage

        return ProviderResponse(
            content=choice.message.content or "",
            model=response.model,
            tokens_in=usage.prompt_tokens     if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=round(elapsed_ms, 1),
            request_id=getattr(response, "id", None),
        )

    async def health_check(self) -> ProviderHealth:
        if not _GROQ_AVAILABLE:
            return ProviderHealth(
                provider_name=self._PROVIDER_NAME,
                model=self._DEFAULT_MODEL,
                status="down",
                avg_latency_ms=None,
                error_rate=1.0,
                is_configured=False,
                last_error="groq SDK not installed",
            )

        if not self._api_key:
            return ProviderHealth(
                provider_name=self._PROVIDER_NAME,
                model=self._DEFAULT_MODEL,
                status="unconfigured",
                avg_latency_ms=None,
                error_rate=0.0,
                is_configured=False,
                last_error="GROQ_API_KEY not set",
            )

        # Lightweight connectivity ping using a minimal prompt
        try:
            t0 = time.monotonic()
            await self._client.chat.completions.create(  # type: ignore[union-attr]
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": '{"ping": true}'}],
                max_tokens=10,
                response_format={"type": "json_object"},
            )
            latency = (time.monotonic() - t0) * 1000
            return ProviderHealth(
                provider_name=self._PROVIDER_NAME,
                model=self._DEFAULT_MODEL,
                status="ok",
                avg_latency_ms=round(latency, 1),
                error_rate=0.0,
                is_configured=True,
            )
        except Exception as exc:
            return ProviderHealth(
                provider_name=self._PROVIDER_NAME,
                model=self._DEFAULT_MODEL,
                status="degraded",
                avg_latency_ms=None,
                error_rate=1.0,
                is_configured=True,
                last_error=str(exc)[:200],
            )

    @property
    def provider_name(self) -> str:
        return self._PROVIDER_NAME

    @property
    def provider_version(self) -> str:
        return self._PROVIDER_VERSION

    @property
    def default_model(self) -> str:
        return self._DEFAULT_MODEL

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key) and _GROQ_AVAILABLE
