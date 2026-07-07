import pytest
from market_intel.models.enums import ImpactLevel, EventStatus, EventCategory
from market_intel.models.event import MFIPEvent
from ..validator.schema_validator import validate_and_build_events


def test_validate_valid_event(sample_raw_event):
    events = validate_and_build_events([sample_raw_event], source_week="thisweek")
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, MFIPEvent)
    assert e.currency == "USD"
    assert e.country == "US"
    assert e.impact == ImpactLevel.HIGH
    assert e.is_high_impact is True
    assert e.status == EventStatus.SCHEDULED
    assert e.actual is None
    assert e.provider == "forex_factory"
    assert e.category == EventCategory.EMPLOYMENT


def test_validate_released_event(sample_raw_event_released):
    events = validate_and_build_events([sample_raw_event_released], source_week="thisweek")
    e = events[0]
    assert e.status == EventStatus.RELEASED
    assert e.actual == "206K"


def test_validate_empty_list():
    assert validate_and_build_events([], source_week="thisweek") == []


def test_validate_skips_bad_keeps_good(sample_raw_event):
    events = validate_and_build_events([{}, sample_raw_event], source_week="thisweek")
    # The good event survives even when the bad one fails
    assert len(events) == 1


def test_speech_classification():
    raw = {
        "title": "Fed Chair Powell Speaks",
        "country": "USD", "date": "Jul 07, 2026", "time": "2:00pm",
        "impact": "High", "forecast": "", "previous": "", "actual": "",
    }
    events = validate_and_build_events([raw], source_week="thisweek")
    assert events[0].is_speech is True
    assert events[0].category == EventCategory.SPEECH


def test_event_id_is_deterministic(sample_raw_event):
    e1 = validate_and_build_events([sample_raw_event], "thisweek")[0]
    e2 = validate_and_build_events([sample_raw_event], "nextweek")[0]
    # source_week difference is in metadata only — event_id should be identical
    assert e1.event_id == e2.event_id
    assert e1.provider_event_id == e2.provider_event_id
