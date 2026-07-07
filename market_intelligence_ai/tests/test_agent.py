"""Tests for MarketIntelligenceAgent — analysis, caching, fallback behaviour."""
from __future__ import annotations

import pytest

from market_intelligence_ai.market_context_compiler.compiler import MarketContextCompiler
from market_intelligence_ai.models.enums import MarketBias, RiskLevel
from market_intelligence_ai.schema.market_intelligence import MarketIntelligenceOutput
from market_intelligence_ai.tests.conftest import make_event_trigger, make_eie_snapshot


@pytest.mark.asyncio
async def test_agent_returns_market_intelligence_output(agent, event_payload):
    result = await agent.analyze(event_payload)
    assert isinstance(result, MarketIntelligenceOutput)


@pytest.mark.asyncio
async def test_agent_returns_bullish_bias(agent, event_payload):
    result = await agent.analyze(event_payload)
    assert result.market_bias == MarketBias.BULLISH
    assert result.is_fallback is False


@pytest.mark.asyncio
async def test_agent_result_has_all_required_fields(agent, event_payload):
    result = await agent.analyze(event_payload)
    assert result.market_bias is not None
    assert result.risk_level is not None
    assert result.expected_duration is not None
    assert result.market_summary != ""
    assert isinstance(result.affected_currencies, list)
    assert result.timestamp is not None


@pytest.mark.asyncio
async def test_agent_caches_result(agent, event_payload):
    result1 = await agent.analyze(event_payload)
    result2 = await agent.analyze(event_payload)
    assert result2.cache_hit is True
    assert result1.market_bias == result2.market_bias


@pytest.mark.asyncio
async def test_agent_does_not_cache_fallback(event_payload):
    """Fallback results must not be stored in cache."""
    from market_intelligence_ai.ai_gateway.gateway import AIGateway
    from market_intelligence_ai.providers.mock_provider import MockProvider
    from market_intelligence_ai.ai_cache.cache import AICache
    from market_intelligence_ai.agent.market_agent import MarketIntelligenceAgent

    failing_provider = MockProvider(raise_on_call=True)
    gateway  = AIGateway(failing_provider)
    cache    = AICache()
    bad_agent = MarketIntelligenceAgent(gateway=gateway, cache=cache)

    result = await bad_agent.analyze(event_payload)
    assert result.is_fallback is True

    key = MarketContextCompiler.cache_key(event_payload)
    cached = await cache.get(key)
    assert cached is None   # fallback must NOT be cached


@pytest.mark.asyncio
async def test_agent_analyze_headline(agent, headline_payload):
    result = await agent.analyze(headline_payload)
    assert isinstance(result, MarketIntelligenceOutput)
    assert result.is_fallback is False


@pytest.mark.asyncio
async def test_agent_fallback_on_provider_failure(failing_provider, context_compiler, fresh_cache, event_payload):
    from market_intelligence_ai.ai_gateway.gateway import AIGateway
    from market_intelligence_ai.agent.market_agent import MarketIntelligenceAgent

    gateway = AIGateway(failing_provider)
    bad_agent = MarketIntelligenceAgent(gateway=gateway, cache=fresh_cache)

    result = await bad_agent.analyze(event_payload)
    assert result.is_fallback is True
    assert result.market_bias == MarketBias.UNCERTAIN
    assert result.risk_level == RiskLevel.MEDIUM
