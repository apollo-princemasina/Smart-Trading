"""
MIA test fixtures.

All tests use MockProvider — no Groq API key required.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from market_intelligence_ai.ai_cache.cache import AICache
from market_intelligence_ai.ai_gateway.gateway import AIGateway
from market_intelligence_ai.agent.market_agent import MarketIntelligenceAgent
from market_intelligence_ai.market_context_compiler.compiler import MarketContextCompiler
from market_intelligence_ai.market_context_compiler.context_models import (
    ContextPayload,
    EventTrigger,
    HeadlineTrigger,
    EIESnapshot,
)
from market_intelligence_ai.models.enums import AnalysisType
from market_intelligence_ai.providers.mock_provider import MockProvider


# ── Provider / Gateway / Agent ────────────────────────────────────────────────

@pytest.fixture
def mock_provider():
    return MockProvider(bias="BULLISH")


@pytest.fixture
def failing_provider():
    return MockProvider(raise_on_call=True)


@pytest.fixture
def mock_gateway(mock_provider):
    return AIGateway(mock_provider)


@pytest.fixture
def fresh_cache():
    return AICache()


@pytest.fixture
def context_compiler():
    return MarketContextCompiler()


@pytest.fixture
def agent(mock_gateway, context_compiler, fresh_cache):
    return MarketIntelligenceAgent(
        gateway          = mock_gateway,
        context_compiler = context_compiler,
        cache            = fresh_cache,
    )


# ── Trigger factories ─────────────────────────────────────────────────────────

def make_event_trigger(
    event_id:           str = "NFP_2026_01",
    title:              str = "Non-Farm Payrolls",
    currency:           str = "USD",
    importance:         str = "HIGH",
    actual:             str = "256K",
    forecast:           str = "220K",
    previous:           str = "199K",
    surprise_class:     str = "LARGE",
    surprise_direction: str = "BEAT",
    economic_direction: str = "BULLISH",
) -> EventTrigger:
    return EventTrigger(
        event_id           = event_id,
        title              = title,
        currency           = currency,
        timestamp          = datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc),
        importance         = importance,
        actual             = actual,
        forecast           = forecast,
        previous           = previous,
        surprise_class     = surprise_class,
        surprise_direction = surprise_direction,
        economic_direction = economic_direction,
    )


def make_headline_trigger(
    headline_id:         str = "HL_001",
    headline:            str = "Fed signals further hikes as inflation remains elevated",
    source:              str = "Reuters",
    affected_currencies: list = None,
) -> HeadlineTrigger:
    return HeadlineTrigger(
        headline_id         = headline_id,
        headline            = headline,
        source              = source,
        timestamp           = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
        affected_currencies = affected_currencies or ["USD"],
    )


def make_eie_snapshot(
    dominant_directions: dict = None,
    execution_risk:      float = 25.0,
    execution_readiness: float = 70.0,
) -> EIESnapshot:
    return EIESnapshot(
        dominant_directions  = dominant_directions or {"USD": "BULLISH"},
        active_events        = [],
        upcoming_high_impact = [],
        execution_risk       = execution_risk,
        execution_readiness  = execution_readiness,
        snapshot_at          = datetime.now(timezone.utc),
    )


@pytest.fixture
def event_trigger():
    return make_event_trigger()


@pytest.fixture
def headline_trigger():
    return make_headline_trigger()


@pytest.fixture
def eie_snapshot():
    return make_eie_snapshot()


@pytest.fixture
def event_payload(context_compiler, event_trigger, eie_snapshot):
    return context_compiler.build_for_event(event_trigger, eie_snapshot)


@pytest.fixture
def headline_payload(context_compiler, headline_trigger, eie_snapshot):
    return context_compiler.build_for_headline(headline_trigger, eie_snapshot)
