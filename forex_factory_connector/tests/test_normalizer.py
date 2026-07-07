from datetime import timezone
import pytest
from ..normalizer.time_normalizer import parse_event_datetime
from ..normalizer.impact_normalizer import normalize_impact
from ..normalizer.value_normalizer import normalize_value
from ..models.enums import ImpactLevel


def test_parse_datetime_to_utc():
    result = parse_event_datetime("Jul 04, 2026", "8:30am")
    assert result is not None
    assert result.tzinfo == timezone.utc
    # 8:30am EDT (UTC-4) -> 12:30 UTC
    assert result.hour == 12
    assert result.minute == 30


def test_parse_all_day_returns_none():
    assert parse_event_datetime("Jul 04, 2026", "All Day") is None
    assert parse_event_datetime("Jul 04, 2026", "") is None


def test_impact_mapping():
    assert normalize_impact("High")   == ImpactLevel.HIGH
    assert normalize_impact("Medium") == ImpactLevel.MEDIUM
    assert normalize_impact("Low")    == ImpactLevel.LOW
    assert normalize_impact("Holiday")== ImpactLevel.HOLIDAY
    assert normalize_impact("")       == ImpactLevel.NON_ECONOMIC
    assert normalize_impact("UNKNOWN")== ImpactLevel.NON_ECONOMIC


def test_value_normalizer():
    assert normalize_value("")      is None
    assert normalize_value(None)    is None
    assert normalize_value("  ")    is None
    assert normalize_value("0.2%")  == "0.2%"
    assert normalize_value(" 1.2B") == "1.2B"
