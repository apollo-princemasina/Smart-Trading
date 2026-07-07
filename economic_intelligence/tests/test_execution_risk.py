"""Unit tests for the ExecutionRiskCalculator."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from economic_intelligence.execution_risk.calculator import ExecutionRiskCalculator
from economic_intelligence.intelligence_models.models import EconomicIntelligenceReport
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.direction_engine.models import EconomicDirection
from economic_intelligence.surprise_engine.models import SurpriseClass, SurpriseDirection
from market_intel.models.enums import ImpactLevel


def _make_report(
    is_released: bool = True,
    importance: ImpactLevel = ImpactLevel.HIGH,
    remaining_influence: float = 50.0,
    time_to_event: float | None = None,
    surprise_class: SurpriseClass = SurpriseClass.NONE,
    economic_direction: EconomicDirection = EconomicDirection.NEUTRAL,
    event_age_hours: float | None = None,
) -> EconomicIntelligenceReport:
    now = datetime.now(timezone.utc)
    return EconomicIntelligenceReport(
        report_id="test_r",
        event_id="test_e",
        generated_at=now,
        event_title="US Non-Farm Employment Change",
        currency="USD",
        country="US",
        timestamp_utc=now,
        is_released=is_released,
        importance=importance,
        event_type=EventType.EMPLOYMENT,
        impact_score=85.0,
        surprise=None,
        pct_surprise=None,
        surprise_class=surprise_class,
        surprise_direction=SurpriseDirection.IN_LINE,
        economic_direction=economic_direction,
        direction_confidence=0.8,
        direction_rationale="test",
        remaining_influence=remaining_influence,
        event_age_hours=event_age_hours,
        time_to_event=time_to_event,
        execution_risk=0.0,
        execution_readiness=0.0,
        confidence=0.8,
        last_updated=now,
    )


def test_holiday_returns_maximum_risk():
    ctx = ExecutionRiskCalculator.compute(
        upcoming_high=[], active_events=[], is_market_open=False, is_holiday=True
    )
    assert ctx.execution_risk >= 90.0
    assert ctx.execution_readiness <= 10.0


def test_market_closed_returns_high_risk():
    ctx = ExecutionRiskCalculator.compute(
        upcoming_high=[], active_events=[], is_market_open=False, is_holiday=False
    )
    assert ctx.execution_risk >= 70.0


def test_imminent_event_within_5_min_is_very_high_risk():
    upcoming = [_make_report(is_released=False, time_to_event=3.0)]
    ctx = ExecutionRiskCalculator.compute(
        upcoming_high=upcoming, active_events=[], is_market_open=True, is_holiday=False
    )
    assert ctx.execution_risk >= 45.0


def test_event_in_90_min_is_low_risk():
    upcoming = [_make_report(is_released=False, time_to_event=90.0)]
    ctx = ExecutionRiskCalculator.compute(
        upcoming_high=upcoming, active_events=[], is_market_open=True, is_holiday=False
    )
    assert ctx.execution_risk <= 15.0


def test_no_events_is_minimal_risk():
    ctx = ExecutionRiskCalculator.compute(
        upcoming_high=[], active_events=[], is_market_open=True, is_holiday=False
    )
    assert ctx.execution_risk <= 10.0
    assert ctx.execution_readiness >= 90.0


def test_high_remaining_influence_adds_risk():
    """Strong remaining influence from a recently released event raises risk."""
    active = [_make_report(remaining_influence=90.0)]
    ctx_high_infl = ExecutionRiskCalculator.compute(
        upcoming_high=[], active_events=active, is_market_open=True, is_holiday=False
    )
    ctx_no_infl = ExecutionRiskCalculator.compute(
        upcoming_high=[], active_events=[], is_market_open=True, is_holiday=False
    )
    assert ctx_high_infl.execution_risk > ctx_no_infl.execution_risk


def test_clustered_events_raise_risk():
    """Multiple HIGH events within 30 minutes → cluster risk."""
    upcoming = [
        _make_report(is_released=False, time_to_event=20.0),
        _make_report(is_released=False, time_to_event=25.0),
        _make_report(is_released=False, time_to_event=28.0),
    ]
    ctx = ExecutionRiskCalculator.compute(
        upcoming_high=upcoming, active_events=[], is_market_open=True, is_holiday=False
    )
    assert ctx.execution_risk >= 30.0


def test_post_release_strong_signal_boosts_readiness():
    """An extreme surprise just released (< 30 min ago) should boost readiness."""
    recent_strong = [_make_report(
        is_released=True,
        remaining_influence=80.0,
        surprise_class=SurpriseClass.EXTREME,
        economic_direction=EconomicDirection.BULLISH,
        event_age_hours=0.2,   # 12 minutes ago
    )]
    ctx_with    = ExecutionRiskCalculator.compute(
        upcoming_high=[], active_events=recent_strong, is_market_open=True, is_holiday=False
    )
    ctx_without = ExecutionRiskCalculator.compute(
        upcoming_high=[], active_events=[], is_market_open=True, is_holiday=False
    )
    assert ctx_with.execution_readiness >= ctx_without.execution_readiness


def test_context_counts_correct():
    upcoming = [_make_report(is_released=False, time_to_event=45.0)] * 3
    active   = [_make_report(is_released=True,  remaining_influence=40.0)] * 2
    ctx = ExecutionRiskCalculator.compute(
        upcoming_high=upcoming, active_events=active, is_market_open=True, is_holiday=False
    )
    assert ctx.upcoming_event_count == 3
    assert ctx.active_event_count   == 2
