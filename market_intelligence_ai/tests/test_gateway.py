"""Tests for AIGateway — retry logic, circuit breaker, fallback behaviour."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from market_intelligence_ai.ai_gateway.gateway import AIGateway
from market_intelligence_ai.providers.base import ProviderError, ProviderRateLimitError
from market_intelligence_ai.providers.mock_provider import MockProvider
from market_intelligence_ai.schema.market_intelligence import MarketIntelligenceOutput
from market_intelligence_ai.models.enums import MarketBias, Importance, TimeHorizon, RiskLevel

_FALLBACK = {
    "market_bias":               MarketBias.UNCERTAIN,
    "affected_currencies":       ["USD"],
    "importance":                Importance.LOW,
    "confidence":                0.0,
    "expected_duration":         TimeHorizon.SHORT_TERM,
    "supports_existing_bias":    False,
    "contradicts_existing_bias": False,
    "risk_level":                RiskLevel.MEDIUM,
    "execution_warning":         "unavailable",
    "market_summary":            "Unavailable.",
    "provider":                  "test",
    "timestamp":                 datetime.now(timezone.utc),
    "latency_ms":                0.0,
}


@pytest.mark.asyncio
async def test_gateway_returns_valid_result(mock_gateway):
    result = await mock_gateway.complete(
        system_prompt = "system",
        user_prompt   = "user",
        model         = "test",
        schema        = MarketIntelligenceOutput,
        fallback_data = _FALLBACK,
    )
    assert isinstance(result, MarketIntelligenceOutput)
    assert result.is_fallback is False
    assert result.market_bias == MarketBias.BULLISH


@pytest.mark.asyncio
async def test_gateway_returns_fallback_on_provider_failure():
    provider = MockProvider(raise_on_call=True)
    gateway  = AIGateway(provider)

    result = await gateway.complete(
        system_prompt = "system",
        user_prompt   = "user",
        model         = "test",
        schema        = MarketIntelligenceOutput,
        fallback_data = _FALLBACK,
        max_retries   = 0,
    )
    assert result.is_fallback is True
    assert result.market_bias == MarketBias.UNCERTAIN


@pytest.mark.asyncio
async def test_gateway_circuit_opens_after_threshold():
    provider = MockProvider(raise_on_call=True)
    gateway  = AIGateway(provider)

    # Trip the circuit (default threshold = 5)
    for _ in range(5):
        await gateway.complete("s", "u", "m", MarketIntelligenceOutput, _FALLBACK, max_retries=0)

    assert gateway.circuit_state == "OPEN"


@pytest.mark.asyncio
async def test_gateway_returns_fallback_when_circuit_open():
    provider = MockProvider(raise_on_call=True)
    gateway  = AIGateway(provider)

    # Trip the circuit
    for _ in range(5):
        await gateway.complete("s", "u", "m", MarketIntelligenceOutput, _FALLBACK, max_retries=0)

    assert gateway.circuit_state == "OPEN"

    # Even with a good provider, the circuit blocks calls
    good_provider = MockProvider(bias="BULLISH")
    gateway._provider = good_provider

    result = await gateway.complete("s", "u", "m", MarketIntelligenceOutput, _FALLBACK)
    assert result.is_fallback is True


@pytest.mark.asyncio
async def test_gateway_tracks_metrics(mock_gateway):
    await mock_gateway.complete("s", "u", "m", MarketIntelligenceOutput, _FALLBACK)
    metrics = mock_gateway.metrics.snapshot()
    assert metrics["total_requests"] == 1
    assert metrics["provider_calls"] == 1
    assert metrics["failed_requests"] == 0
