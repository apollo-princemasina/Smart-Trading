"""
MarketIntelligenceAgent — the single autonomous AI analyst.

Receives a ContextPayload assembled by the MarketContextCompiler, formats it
into a user message, calls the AI provider through the AIGateway, validates the
structured response, and returns a MarketIntelligenceOutput.

There is exactly ONE agent in the MIA layer. It does not know whether it is
analysing a CPI release, an NFP surprise, a central bank headline, or any other
event type. It applies five institutional reasoning perspectives (defined in the
system prompt) and reasons autonomously from the context it receives.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from market_intelligence_ai.ai_gateway.gateway import AIGateway
from market_intelligence_ai.ai_cache.cache import AICache, ai_cache
from market_intelligence_ai.market_context_compiler.compiler import MarketContextCompiler
from market_intelligence_ai.market_context_compiler.context_models import ContextPayload
from market_intelligence_ai.schema.system_prompt import SYSTEM_PROMPT
from market_intelligence_ai.schema.market_intelligence import MarketIntelligenceOutput
from market_intelligence_ai.models.enums import MarketBias, Importance, TimeHorizon, RiskLevel, AnalysisType
from market_intelligence_ai.utils.config import mia_config
from market_intelligence_ai.utils.logger import logger


class MarketIntelligenceAgent:
    """
    Single autonomous institutional market analyst.

    Design contract:
    - Accepts any ContextPayload (event, headline, or combined)
    - Returns a consistent MarketIntelligenceOutput regardless of input type
    - Never selects different prompts for different event types
    - Applies five reasoning perspectives instructed in the system prompt
    - Cache-aware: checks AICache before calling the provider
    - Graceful degradation: returns fallback with is_fallback=True on failure
    """

    def __init__(
        self,
        gateway:          AIGateway,
        context_compiler: Optional[MarketContextCompiler] = None,
        cache:            Optional[AICache] = None,
    ) -> None:
        self._gateway          = gateway
        self._context_compiler = context_compiler or MarketContextCompiler()
        self._cache            = cache or ai_cache

    async def analyze(self, payload: ContextPayload) -> MarketIntelligenceOutput:
        """
        Analyse a ContextPayload and return structured market intelligence.

        The agent:
        1. Checks cache — returns immediately on hit
        2. Formats the context into a user message via MarketContextCompiler
        3. Calls the AIGateway (which handles retry, circuit breaker, validation)
        4. Stamps provider metadata
        5. Caches the result (fallbacks are never cached)
        """
        payload.validate()
        cache_key = MarketContextCompiler.cache_key(
            payload,
            provider_version=mia_config.PROVIDER_VERSION,
        )

        cached = await self._cache.get(cache_key)
        if cached is not None:
            cached.cache_hit = True
            logger.debug("MarketIntelligenceAgent: cache HIT for key {}…", cache_key[:16])
            return cached

        logger.debug(
            "MarketIntelligenceAgent: cache MISS — calling provider for {} {}",
            payload.analysis_type.value, payload.primary_currency,
        )

        user_message = self._context_compiler.format_as_user_message(payload)

        fallback_data = {
            "market_bias":               MarketBias.UNCERTAIN,
            "affected_currencies":       [payload.primary_currency],
            "importance":                Importance.LOW,
            "confidence":                0.0,
            "expected_duration":         TimeHorizon.SHORT_TERM,
            "supports_existing_bias":    False,
            "contradicts_existing_bias": False,
            "risk_level":                RiskLevel.MEDIUM,
            "execution_warning":         "Analysis unavailable — AI provider error.",
            "market_summary":            "Market intelligence temporarily unavailable.",
            "provider":                  self._gateway.provider.provider_name,
            "timestamp":                 datetime.now(timezone.utc),
            "latency_ms":                0.0,
        }

        result: MarketIntelligenceOutput = await self._gateway.complete(
            system_prompt = SYSTEM_PROMPT,
            user_prompt   = user_message,
            model         = mia_config.MIA_ANALYSIS_MODEL,
            schema        = MarketIntelligenceOutput,
            fallback_data = fallback_data,
            temperature   = mia_config.MIA_TEMPERATURE,
            max_tokens    = mia_config.MIA_MAX_TOKENS,
        )

        if not result.timestamp:
            result.timestamp = datetime.now(timezone.utc)

        if not result.provider:
            result.provider = f"{self._gateway.provider.provider_name}:{mia_config.MIA_ANALYSIS_MODEL}"

        if not result.is_fallback:
            ttl = self._pick_ttl(payload)
            await self._cache.set(cache_key, result, ttl_seconds=ttl)

        logger.info(
            "MarketIntelligenceAgent: {} {} → {} risk={} (conf={:.2f}, fallback={}, cached={})",
            payload.analysis_type.value,
            payload.primary_currency,
            result.market_bias,
            result.risk_level,
            result.confidence,
            result.is_fallback,
            not result.is_fallback,
        )
        return result

    def _pick_ttl(self, payload: ContextPayload) -> int:
        if payload.analysis_type == AnalysisType.EVENT:
            return mia_config.MIA_EVENT_CACHE_TTL_S
        elif payload.analysis_type == AnalysisType.HEADLINE:
            return mia_config.MIA_HEADLINE_CACHE_TTL_S
        elif payload.analysis_type == AnalysisType.GENERAL:
            return 1800  # 30 min — matches the general narrative cycle interval
        return mia_config.MIA_COMBINED_CACHE_TTL_S
