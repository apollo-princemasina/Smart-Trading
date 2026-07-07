"""
MockProvider — deterministic test provider for use in unit and integration tests.

Returns pre-configured MarketIntelligenceOutput-compatible JSON responses
without any network calls.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from market_intelligence_ai.providers.base import (
    MarketIntelligenceProvider,
    ProviderError,
    ProviderHealth,
    ProviderResponse,
)

# ── Default mock response — matches MarketIntelligenceOutput schema exactly ───

_DEFAULT_RESPONSE: dict = {
    "market_bias":               "BULLISH",
    "affected_currencies":       ["USD"],
    "importance":                "HIGH",
    "confidence":                0.78,
    "expected_duration":         "SHORT_TERM",
    "supports_existing_bias":    True,
    "contradicts_existing_bias": False,
    "risk_level":                "LOW",
    "execution_warning":         None,
    "market_summary": (
        "Strong US labor market data reinforces the bullish USD thesis. "
        "Markets are pricing in a higher-for-longer Fed stance, supporting USD demand "
        "against major pairs. Institutional positioning remains net long USD."
    ),
}

_BEARISH_RESPONSE: dict = {
    **_DEFAULT_RESPONSE,
    "market_bias":               "BEARISH",
    "risk_level":                "MEDIUM",
    "supports_existing_bias":    False,
    "contradicts_existing_bias": True,
    "market_summary": (
        "Weaker-than-expected data undermines the bullish narrative. "
        "Markets are reassessing the Fed's policy path, with downside pressure "
        "building on USD as rate cut expectations are repriced."
    ),
}

_NEUTRAL_RESPONSE: dict = {
    **_DEFAULT_RESPONSE,
    "market_bias":               "NEUTRAL",
    "confidence":                0.45,
    "risk_level":                "LOW",
    "supports_existing_bias":    False,
    "contradicts_existing_bias": False,
    "market_summary": (
        "In-line data release provides no significant directional catalyst. "
        "Market impact expected to be limited with participants remaining focused "
        "on upcoming high-impact events."
    ),
}


class MockProvider(MarketIntelligenceProvider):
    """
    Test/development provider that returns deterministic mock responses.

    Configure `raise_on_call` to simulate provider failures in tests.
    Configure `response_override` to return a specific response for all calls.
    Configure `bias` to control the directional bias of responses ("BULLISH" | "BEARISH" | "NEUTRAL").
    """

    def __init__(
        self,
        raise_on_call:     bool = False,
        response_override: Optional[dict] = None,
        bias:              str = "BULLISH",
        latency_ms:        float = 50.0,
    ) -> None:
        self._raise_on_call     = raise_on_call
        self._response_override = response_override
        self._bias              = bias.upper()
        self._latency_ms        = latency_ms
        self._call_count        = 0

    async def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        model:         str,
        temperature:   float,
        max_tokens:    int,
    ) -> ProviderResponse:
        self._call_count += 1

        if self._raise_on_call:
            raise ProviderError("MockProvider configured to raise on call")

        await asyncio.sleep(0.01)  # simulate minimal async latency

        content = self._pick_response(user_prompt)

        return ProviderResponse(
            content    = json.dumps(content),
            model      = model,
            tokens_in  = len(user_prompt) // 4,
            tokens_out = len(json.dumps(content)) // 4,
            latency_ms = self._latency_ms,
            request_id = f"mock_{self._call_count}",
        )

    def _pick_response(self, user_prompt: str) -> dict:
        if self._response_override is not None:
            return dict(self._response_override)
        if self._bias == "BEARISH":
            return dict(_BEARISH_RESPONSE)
        if self._bias == "NEUTRAL":
            return dict(_NEUTRAL_RESPONSE)
        return dict(_DEFAULT_RESPONSE)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            provider_name  = "mock",
            model          = "mock-model",
            status         = "ok",
            avg_latency_ms = self._latency_ms,
            error_rate     = 0.0,
            is_configured  = True,
        )

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def provider_version(self) -> str:
        return "mock_v1"

    @property
    def default_model(self) -> str:
        return "mock-model"

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count
