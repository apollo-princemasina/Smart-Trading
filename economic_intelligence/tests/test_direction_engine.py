"""Unit tests for the DirectionRuleEngine."""
from __future__ import annotations

import pytest

from economic_intelligence.direction_engine.rule_engine import DirectionRuleEngine
from economic_intelligence.direction_engine.models import EconomicDirection
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.surprise_engine.models import (
    SurpriseResult, SurpriseClass, SurpriseDirection
)
from market_intel.models.enums import EventStatus
from economic_intelligence.tests.conftest import make_event


def _beat(raw=21000.0, pct=11.4, cls=SurpriseClass.SMALL) -> SurpriseResult:
    return SurpriseResult(
        actual_value=206_000.0, forecast_value=185_000.0,
        raw_surprise=raw, pct_surprise=pct,
        direction=SurpriseDirection.BEAT, surprise_class=cls,
    )


def _miss(raw=-21000.0, pct=-11.4, cls=SurpriseClass.SMALL) -> SurpriseResult:
    return SurpriseResult(
        actual_value=164_000.0, forecast_value=185_000.0,
        raw_surprise=raw, pct_surprise=pct,
        direction=SurpriseDirection.MISS, surprise_class=cls,
    )


def _inline() -> SurpriseResult:
    return SurpriseResult(
        actual_value=185_000.0, forecast_value=185_000.0,
        raw_surprise=0.0, pct_surprise=0.0,
        direction=SurpriseDirection.IN_LINE, surprise_class=SurpriseClass.NONE,
    )


# ── EMPLOYMENT ────────────────────────────────────────────────────────────────

def test_employment_beat_is_bullish():
    event = make_event("US Non-Farm Employment Change")
    signal = DirectionRuleEngine.resolve(event, EventType.EMPLOYMENT, _beat())
    assert signal.direction == EconomicDirection.BULLISH
    assert signal.confidence > 0.7


def test_employment_miss_is_bearish():
    event = make_event("US Non-Farm Employment Change")
    signal = DirectionRuleEngine.resolve(event, EventType.EMPLOYMENT, _miss())
    assert signal.direction == EconomicDirection.BEARISH


# ── UNEMPLOYMENT (higher = bearish) ──────────────────────────────────────────

def test_unemployment_beat_is_bearish():
    """Higher unemployment rate is a beat of the absolute number → bearish."""
    event = make_event("German Unemployment Rate", currency="EUR")
    signal = DirectionRuleEngine.resolve(event, EventType.UNEMPLOYMENT, _beat())
    assert signal.direction == EconomicDirection.BEARISH


def test_unemployment_miss_is_bullish():
    """Lower unemployment rate = miss of the number = good = bullish."""
    event = make_event("German Unemployment Rate", currency="EUR")
    signal = DirectionRuleEngine.resolve(event, EventType.UNEMPLOYMENT, _miss())
    assert signal.direction == EconomicDirection.BULLISH


# ── INFLATION ─────────────────────────────────────────────────────────────────

def test_inflation_beat_is_bullish():
    event = make_event("US CPI m/m", forecast="0.2%", actual="0.4%")
    signal = DirectionRuleEngine.resolve(event, EventType.INFLATION, _beat())
    assert signal.direction == EconomicDirection.BULLISH


def test_inflation_miss_is_bearish():
    event = make_event("US CPI m/m", forecast="0.2%", actual="0.1%")
    signal = DirectionRuleEngine.resolve(event, EventType.INFLATION, _miss())
    assert signal.direction == EconomicDirection.BEARISH


# ── INTEREST RATE ─────────────────────────────────────────────────────────────

def test_interest_rate_beat_is_bullish():
    event = make_event("FOMC Statement", forecast="5.25%", actual="5.50%")
    signal = DirectionRuleEngine.resolve(event, EventType.INTEREST_RATE, _beat())
    assert signal.direction == EconomicDirection.BULLISH
    assert signal.confidence >= 0.85


# ── IN_LINE ───────────────────────────────────────────────────────────────────

def test_inline_produces_neutral():
    event = make_event("US Non-Farm Employment Change")
    signal = DirectionRuleEngine.resolve(event, EventType.EMPLOYMENT, _inline())
    assert signal.direction == EconomicDirection.NEUTRAL


# ── SCHEDULED ────────────────────────────────────────────────────────────────

def test_scheduled_event_returns_uncertain():
    event = make_event("US Non-Farm Employment Change", status=EventStatus.SCHEDULED, actual=None)
    signal = DirectionRuleEngine.resolve(event, EventType.EMPLOYMENT, surprise=None)
    assert signal.direction == EconomicDirection.UNCERTAIN
    assert signal.confidence == 0.0


# ── SURPRISE MAGNITUDE BOOSTS CONFIDENCE ─────────────────────────────────────

def test_extreme_surprise_boosts_confidence():
    event = make_event("US CPI m/m")
    small_beat = _beat(cls=SurpriseClass.SMALL)
    extreme_beat = _beat(cls=SurpriseClass.EXTREME)

    sig_small   = DirectionRuleEngine.resolve(event, EventType.INFLATION, small_beat)
    sig_extreme = DirectionRuleEngine.resolve(event, EventType.INFLATION, extreme_beat)

    assert sig_extreme.confidence > sig_small.confidence


# ── UNCERTAIN TYPES ───────────────────────────────────────────────────────────

def test_political_event_is_uncertain():
    event = make_event("US Presidential Election")
    signal = DirectionRuleEngine.resolve(event, EventType.POLITICAL, _beat())
    assert signal.direction == EconomicDirection.UNCERTAIN


def test_holiday_event_is_uncertain():
    event = make_event("Bank Holiday")
    signal = DirectionRuleEngine.resolve(event, EventType.HOLIDAY, None)
    assert signal.direction == EconomicDirection.UNCERTAIN
