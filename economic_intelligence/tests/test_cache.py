"""Unit tests for the IntelligenceCache."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio

from economic_intelligence.intelligence_cache.cache import IntelligenceCache
from economic_intelligence.intelligence_models.models import EconomicIntelligenceReport
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.direction_engine.models import EconomicDirection
from economic_intelligence.surprise_engine.models import SurpriseClass, SurpriseDirection
from market_intel.models.enums import ImpactLevel


def _make_report(
    event_id: str = "evt_1",
    currency: str = "USD",
    is_released: bool = True,
    remaining_influence: float = 50.0,
    importance: ImpactLevel = ImpactLevel.HIGH,
    timestamp_offset_hours: float = -1.0,
) -> EconomicIntelligenceReport:
    now = datetime.now(timezone.utc)
    ts = now + timedelta(hours=timestamp_offset_hours)
    return EconomicIntelligenceReport(
        report_id="rpt_" + event_id,
        event_id=event_id,
        generated_at=now,
        event_title="Test Event",
        currency=currency,
        country="US",
        timestamp_utc=ts,
        is_released=is_released,
        importance=importance,
        event_type=EventType.EMPLOYMENT,
        impact_score=80.0,
        surprise=None,
        pct_surprise=None,
        surprise_class=SurpriseClass.NONE,
        surprise_direction=SurpriseDirection.IN_LINE,
        economic_direction=EconomicDirection.BULLISH,
        direction_confidence=0.8,
        direction_rationale="test",
        remaining_influence=remaining_influence,
        event_age_hours=abs(timestamp_offset_hours),
        time_to_event=None if is_released else abs(timestamp_offset_hours) * 60,
        execution_risk=20.0,
        execution_readiness=80.0,
        confidence=0.8,
        last_updated=now,
    )


@pytest.mark.asyncio
async def test_cache_starts_empty():
    cache = IntelligenceCache()
    assert not cache.is_populated
    all_reports = await cache.get_all()
    assert all_reports == []


@pytest.mark.asyncio
async def test_set_and_retrieve_reports():
    cache = IntelligenceCache()
    reports = [_make_report("evt_1"), _make_report("evt_2")]
    await cache.set_reports(reports, context=None)

    all_reports = await cache.get_all()
    assert len(all_reports) == 2
    assert cache.is_populated


@pytest.mark.asyncio
async def test_active_events_filter():
    """Only released events above the active threshold are returned."""
    cache = IntelligenceCache()
    active_report   = _make_report("active",   remaining_influence=60.0, is_released=True)
    inactive_report = _make_report("inactive", remaining_influence=0.5,  is_released=True)
    sched_report    = _make_report("sched",    remaining_influence=80.0, is_released=False)

    await cache.set_reports([active_report, inactive_report, sched_report], context=None)

    active = await cache.get_active()
    ids = {r.event_id for r in active}
    assert "active" in ids
    assert "inactive" not in ids
    assert "sched" not in ids


@pytest.mark.asyncio
async def test_upcoming_events_filter():
    """Only non-released events within the time window are returned."""
    cache = IntelligenceCache()
    # Scheduled event 2 hours in the future
    upcoming = _make_report("upcoming", is_released=False, timestamp_offset_hours=2.0)
    # Released event
    released = _make_report("released", is_released=True,  timestamp_offset_hours=-1.0)
    # Scheduled event far in the future (outside 24h window)
    far_future = _make_report("far", is_released=False, timestamp_offset_hours=100.0)

    await cache.set_reports([upcoming, released, far_future], context=None)

    result = await cache.get_upcoming(limit_hours=24.0)
    ids = {r.event_id for r in result}
    assert "upcoming" in ids
    assert "released" not in ids
    assert "far" not in ids


@pytest.mark.asyncio
async def test_currency_filter():
    cache = IntelligenceCache()
    usd = _make_report("usd_evt", currency="USD")
    eur = _make_report("eur_evt", currency="EUR")
    await cache.set_reports([usd, eur], context=None)

    usd_results = await cache.get_by_currency("USD")
    assert all(r.currency == "USD" for r in usd_results)
    assert len(usd_results) == 1


@pytest.mark.asyncio
async def test_report_update_replaces_same_event():
    """A second call with the same event_id updates (not duplicates) the report."""
    cache = IntelligenceCache()
    original = _make_report("evt_1", remaining_influence=50.0)
    updated  = _make_report("evt_1", remaining_influence=30.0)

    await cache.set_reports([original], context=None)
    await cache.set_reports([updated],  context=None)

    all_reports = await cache.get_all()
    assert len(all_reports) == 1
    assert all_reports[0].remaining_influence == 30.0


@pytest.mark.asyncio
async def test_cycle_count_increments():
    cache = IntelligenceCache()
    assert cache.cycle_count == 0
    await cache.set_reports([], context=None)
    await cache.set_reports([], context=None)
    assert cache.cycle_count == 2
