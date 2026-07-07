"""Unit tests for the DecayCalculator."""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pytest

from economic_intelligence.decay_engine.calculator import DecayCalculator
from economic_intelligence.decay_engine.curves import get_decay_profile
from economic_intelligence.event_classifier.event_types import EventType
from market_intel.models.enums import EventStatus
from economic_intelligence.tests.conftest import make_event


BASE_SCORE = 85.0


def _released_event(hours_ago: float, event_type_title: str = "US Non-Farm Employment Change"):
    return make_event(event_type_title, timestamp_offset_hours=-hours_ago)


def test_at_release_influence_equals_base_score():
    """At t=0 (just released), remaining influence should be close to base_score."""
    event = _released_event(0.0)
    now = event.timestamp_utc
    remaining, age = DecayCalculator.compute(event, EventType.EMPLOYMENT, BASE_SCORE, now)
    assert abs(remaining - BASE_SCORE) < 0.5


def test_at_half_life_influence_is_half():
    """At t=half_life, remaining influence should be ~base_score * 0.5."""
    profile = get_decay_profile(EventType.EMPLOYMENT)
    hours = profile.half_life_hours

    event = _released_event(hours)
    now = datetime.now(timezone.utc)
    remaining, age = DecayCalculator.compute(event, EventType.EMPLOYMENT, BASE_SCORE, now)

    expected = max(profile.min_influence, BASE_SCORE * 0.5)
    assert abs(remaining - expected) < 1.0


def test_fomc_decays_slower_than_pmi():
    """FOMC half-life > PMI half-life → FOMC influence higher after 6h."""
    hours_ago = 6.0
    event_fomc = _released_event(hours_ago, "FOMC Statement")
    event_pmi  = _released_event(hours_ago, "US ISM Manufacturing PMI")
    now = datetime.now(timezone.utc)

    remaining_fomc, _ = DecayCalculator.compute(event_fomc, EventType.INTEREST_RATE, BASE_SCORE, now)
    remaining_pmi,  _ = DecayCalculator.compute(event_pmi,  EventType.PMI,           BASE_SCORE, now)

    assert remaining_fomc > remaining_pmi, (
        f"FOMC ({remaining_fomc:.1f}) should be > PMI ({remaining_pmi:.1f}) after {hours_ago}h"
    )


def test_very_old_event_reaches_minimum():
    """After a very long time, remaining influence should floor at min_influence."""
    profile = get_decay_profile(EventType.PMI)
    hours_ago = 200.0  # way past any reasonable half-life

    event = _released_event(hours_ago, "US ISM Manufacturing PMI")
    now = datetime.now(timezone.utc)
    remaining, _ = DecayCalculator.compute(event, EventType.PMI, BASE_SCORE, now)

    assert abs(remaining - profile.min_influence) < 0.01


def test_scheduled_event_returns_zero():
    event = make_event(
        "US Non-Farm Employment Change",
        status=EventStatus.SCHEDULED,
        actual=None,
        timestamp_offset_hours=2.0,
    )
    remaining, age = DecayCalculator.compute(event, EventType.EMPLOYMENT, BASE_SCORE)
    assert remaining == 0.0
    assert age is None


def test_age_is_correct():
    hours_ago = 3.5
    event = _released_event(hours_ago)
    now = datetime.now(timezone.utc)
    _, age = DecayCalculator.compute(event, EventType.EMPLOYMENT, BASE_SCORE, now)
    assert age is not None
    assert abs(age - hours_ago) < 0.1


def test_future_event_returns_zero():
    """An event timestamped in the future returns zero influence even if released."""
    event = make_event(
        "US Non-Farm Employment Change",
        status=EventStatus.RELEASED,
        timestamp_offset_hours=2.0,  # in the future
    )
    now = datetime.now(timezone.utc)
    remaining, age = DecayCalculator.compute(event, EventType.EMPLOYMENT, BASE_SCORE, now)
    assert remaining == 0.0
    assert age is None
