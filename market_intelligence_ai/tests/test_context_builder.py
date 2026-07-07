"""Tests for MarketContextCompiler — context assembly, completeness, and cache keys."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from market_intelligence_ai.market_context_compiler.compiler import (
    MarketContextCompiler,
    _detect_session,
)
from market_intelligence_ai.market_context_compiler.context_models import EIESnapshot, ContextPayload
from market_intelligence_ai.models.enums import AnalysisType
from market_intelligence_ai.utils.config import mia_config

from market_intelligence_ai.tests.conftest import (
    make_event_trigger,
    make_headline_trigger,
    make_eie_snapshot,
)


# ── Session detection ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("hour,expected", [
    (0,  "ASIA"),
    (4,  "ASIA"),
    (8,  "LONDON"),
    (12, "LONDON"),
    (14, "OVERLAP"),
    (16, "OVERLAP"),
    (17, "NEW_YORK"),
    (20, "NEW_YORK"),
    (22, "ASIA"),
])
def test_session_detection(hour, expected):
    dt = datetime(2026, 1, 1, hour, 0, tzinfo=timezone.utc)
    assert _detect_session(dt) == expected


# ── Event payload assembly ────────────────────────────────────────────────────

def test_build_for_event_sets_type(context_compiler, event_trigger, eie_snapshot):
    payload = context_compiler.build_for_event(event_trigger, eie_snapshot)
    assert payload.analysis_type == AnalysisType.EVENT
    assert payload.event_trigger is event_trigger
    assert payload.headline_trigger is None


def test_build_for_event_primary_currency(context_compiler):
    trigger = make_event_trigger(currency="EUR")
    payload = context_compiler.build_for_event(trigger)
    assert payload.primary_currency == "EUR"


def test_build_for_event_without_eie_succeeds(context_compiler, event_trigger):
    payload = context_compiler.build_for_event(event_trigger, eie_snapshot=None)
    assert isinstance(payload.eie_snapshot, EIESnapshot)


# ── Headline payload assembly ─────────────────────────────────────────────────

def test_build_for_headline_sets_type(context_compiler, headline_trigger, eie_snapshot):
    payload = context_compiler.build_for_headline(headline_trigger, eie_snapshot)
    assert payload.analysis_type == AnalysisType.HEADLINE
    assert payload.headline_trigger is headline_trigger
    assert payload.event_trigger is None


def test_build_for_headline_primary_currency(context_compiler):
    trigger = make_headline_trigger(affected_currencies=["GBP", "USD"])
    payload = context_compiler.build_for_headline(trigger)
    assert payload.primary_currency == "GBP"   # first currency


# ── Payload validation ────────────────────────────────────────────────────────

def test_empty_payload_raises():
    payload = ContextPayload(
        analysis_type    = AnalysisType.EVENT,
        primary_currency = "USD",
        event_trigger    = None,
        headline_trigger = None,
    )
    with pytest.raises(ValueError, match="at least one trigger"):
        payload.validate()


# ── User message formatting ───────────────────────────────────────────────────

def test_format_event_message_contains_key_fields(context_compiler, event_trigger, eie_snapshot):
    payload = context_compiler.build_for_event(event_trigger, eie_snapshot)
    msg = context_compiler.format_as_user_message(payload)

    assert "Non-Farm Payrolls" in msg
    assert "USD" in msg
    assert "256K" in msg        # actual
    assert "220K" in msg        # forecast
    assert "LARGE" in msg       # surprise_class
    assert "BULLISH" in msg     # EIE direction
    assert "ANALYSIS REQUEST" in msg
    assert "five institutional reasoning perspectives" in msg.lower() or "json analysis" in msg.lower()


def test_format_headline_message_contains_key_fields(context_compiler, headline_trigger, eie_snapshot):
    payload = context_compiler.build_for_headline(headline_trigger, eie_snapshot)
    msg = context_compiler.format_as_user_message(payload)

    assert "Fed signals" in msg
    assert "Reuters" in msg
    assert "USD" in msg


def test_format_includes_eie_context(context_compiler, event_trigger):
    snap = make_eie_snapshot(
        dominant_directions = {"USD": "BULLISH", "EUR": "BEARISH"},
        execution_risk      = 42.0,
    )
    payload = context_compiler.build_for_event(event_trigger, snap)
    msg = context_compiler.format_as_user_message(payload)

    assert "BULLISH" in msg
    assert "42" in msg          # execution_risk


# ── Cache key stability ───────────────────────────────────────────────────────

def test_cache_key_deterministic(context_compiler, event_trigger, eie_snapshot):
    p1 = context_compiler.build_for_event(event_trigger, eie_snapshot)
    p2 = context_compiler.build_for_event(event_trigger, eie_snapshot)
    assert MarketContextCompiler.cache_key(p1) == MarketContextCompiler.cache_key(p2)


def test_cache_key_differs_by_event_id(context_compiler, eie_snapshot):
    t1 = make_event_trigger(event_id="EVT_001")
    t2 = make_event_trigger(event_id="EVT_002")
    p1 = context_compiler.build_for_event(t1, eie_snapshot)
    p2 = context_compiler.build_for_event(t2, eie_snapshot)
    assert MarketContextCompiler.cache_key(p1) != MarketContextCompiler.cache_key(p2)


def test_cache_key_differs_by_surprise_class(context_compiler, eie_snapshot):
    t1 = make_event_trigger(surprise_class="NONE")
    t2 = make_event_trigger(surprise_class="EXTREME")
    p1 = context_compiler.build_for_event(t1, eie_snapshot)
    p2 = context_compiler.build_for_event(t2, eie_snapshot)
    assert MarketContextCompiler.cache_key(p1) != MarketContextCompiler.cache_key(p2)


def test_cache_key_includes_provider_version(context_compiler, event_trigger, eie_snapshot):
    """Provider version changes must invalidate cache entries."""
    payload = context_compiler.build_for_event(event_trigger, eie_snapshot)
    key_v1 = MarketContextCompiler.cache_key(payload, provider_version="groq_v1")
    key_v2 = MarketContextCompiler.cache_key(payload, provider_version="groq_v2")
    assert key_v1 != key_v2


def test_cache_key_includes_schema_version(context_compiler, event_trigger, eie_snapshot):
    """Schema version is embedded in the cache key via config."""
    payload = context_compiler.build_for_event(event_trigger, eie_snapshot)
    key = MarketContextCompiler.cache_key(payload)
    assert len(key) == 32  # sha256 hex truncated to 32 chars
