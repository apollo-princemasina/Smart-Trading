"""Unit tests for the ImpactCalculator."""
from __future__ import annotations

import pytest

from economic_intelligence.impact_engine.calculator import ImpactCalculator
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.surprise_engine.models import (
    SurpriseResult, SurpriseClass, SurpriseDirection,
)
from market_intel.models.enums import ImpactLevel
from economic_intelligence.tests.conftest import make_event


def _surprise(cls: SurpriseClass) -> SurpriseResult:
    return SurpriseResult(
        actual_value=1.0, forecast_value=1.0,
        raw_surprise=0.0, pct_surprise=0.0,
        direction=SurpriseDirection.IN_LINE,
        surprise_class=cls,
    )


def test_high_impact_nfp_baseline():
    event = make_event("US Non-Farm Employment Change", impact=ImpactLevel.HIGH)
    score = ImpactCalculator.compute(event, EventType.EMPLOYMENT)
    assert 80.0 <= score <= 95.0


def test_medium_impact_lower_than_high():
    event_high   = make_event("US NFP",   impact=ImpactLevel.HIGH)
    event_medium = make_event("US PMI",   impact=ImpactLevel.MEDIUM)
    score_high   = ImpactCalculator.compute(event_high,   EventType.EMPLOYMENT)
    score_medium = ImpactCalculator.compute(event_medium, EventType.PMI)
    assert score_high > score_medium


def test_low_impact_event():
    event = make_event("US Something Minor", impact=ImpactLevel.LOW)
    score = ImpactCalculator.compute(event, EventType.UNKNOWN)
    assert score <= 15.0


def test_holiday_score_near_zero():
    event = make_event("Bank Holiday", impact=ImpactLevel.HOLIDAY)
    score = ImpactCalculator.compute(event, EventType.HOLIDAY)
    assert score <= 5.0


def test_extreme_surprise_amplifies_score():
    event = make_event("US CPI m/m", impact=ImpactLevel.HIGH)
    score_no_surprise   = ImpactCalculator.compute(event, EventType.INFLATION)
    score_extreme       = ImpactCalculator.compute(event, EventType.INFLATION, _surprise(SurpriseClass.EXTREME))
    assert score_extreme > score_no_surprise


def test_inline_surprise_does_not_change_score():
    event = make_event("US CPI m/m", impact=ImpactLevel.HIGH)
    score_no_surprise  = ImpactCalculator.compute(event, EventType.INFLATION)
    score_inline       = ImpactCalculator.compute(event, EventType.INFLATION, _surprise(SurpriseClass.NONE))
    assert abs(score_no_surprise - score_inline) < 0.01


def test_score_capped_at_100():
    event = make_event("US Interest Rate", impact=ImpactLevel.HIGH)
    score = ImpactCalculator.compute(event, EventType.INTEREST_RATE, _surprise(SurpriseClass.EXTREME))
    assert score <= 100.0


def test_interest_rate_has_highest_weight():
    """Interest rate decisions should produce the highest possible base score."""
    event_rate = make_event("FOMC Statement",      impact=ImpactLevel.HIGH)
    event_pmi  = make_event("US Manufacturing PMI", impact=ImpactLevel.HIGH)
    score_rate = ImpactCalculator.compute(event_rate, EventType.INTEREST_RATE)
    score_pmi  = ImpactCalculator.compute(event_pmi,  EventType.PMI)
    assert score_rate > score_pmi
