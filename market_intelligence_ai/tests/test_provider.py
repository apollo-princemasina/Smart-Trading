"""Tests for provider abstraction and MockProvider."""
from __future__ import annotations

import json

import pytest
import pytest_asyncio

from market_intelligence_ai.providers.base import ProviderError, ProviderResponse
from market_intelligence_ai.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_mock_provider_returns_response():
    provider = MockProvider()
    response = await provider.complete(
        system_prompt = "You are an analyst.",
        user_prompt   = "Analyse this data.",
        model         = "test-model",
        temperature   = 0.1,
        max_tokens    = 512,
    )
    assert isinstance(response, ProviderResponse)
    assert response.model == "test-model"
    assert isinstance(response.content, str)


@pytest.mark.asyncio
async def test_mock_provider_returns_valid_json():
    provider = MockProvider(bias="BULLISH")
    response = await provider.complete(
        system_prompt = "system",
        user_prompt   = "user",
        model         = "m",
        temperature   = 0.1,
        max_tokens    = 512,
    )
    data = json.loads(response.content)
    assert data["market_bias"] == "BULLISH"
    assert "confidence" in data
    assert "summary" in data


@pytest.mark.asyncio
async def test_mock_provider_bearish_bias():
    provider = MockProvider(bias="BEARISH")
    response = await provider.complete("s", "u", "m", 0.1, 512)
    data = json.loads(response.content)
    assert data["market_bias"] == "BEARISH"


@pytest.mark.asyncio
async def test_mock_provider_raise_on_call():
    provider = MockProvider(raise_on_call=True)
    with pytest.raises(ProviderError):
        await provider.complete("s", "u", "m", 0.1, 512)


@pytest.mark.asyncio
async def test_mock_provider_call_count():
    provider = MockProvider()
    assert provider.call_count == 0
    await provider.complete("s", "u", "m", 0.1, 512)
    await provider.complete("s", "u", "m", 0.1, 512)
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_mock_provider_health_check():
    provider = MockProvider()
    health = await provider.health_check()
    assert health.status == "ok"
    assert health.is_configured is True


@pytest.mark.asyncio
async def test_mock_provider_response_override():
    override = {
        "market_bias":              "NEUTRAL",
        "affected_currencies":      ["EUR"],
        "importance":               "MEDIUM",
        "confidence":               0.5,
        "time_horizon":             "MEDIUM_TERM",
        "supports_existing_bias":   False,
        "contradicts_existing_bias": False,
        "execution_warning":        None,
        "summary":                  "Custom override response.",
    }
    provider = MockProvider(response_override=override)
    response = await provider.complete("s", "u", "m", 0.1, 512)
    data = json.loads(response.content)
    assert data["market_bias"] == "NEUTRAL"
    assert data["summary"] == "Custom override response."
