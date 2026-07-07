"""Unit tests for the value parser and SurpriseCalculator."""
from __future__ import annotations

import pytest

from economic_intelligence.surprise_engine.value_parser import parse_economic_value
from economic_intelligence.surprise_engine.calculator import SurpriseCalculator
from economic_intelligence.surprise_engine.models import (
    SurpriseClass,
    SurpriseDirection,
)
from economic_intelligence.event_classifier.event_types import EventType
from market_intel.models.enums import EventStatus
from economic_intelligence.tests.conftest import make_event


# ── Value parser ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("185K",   185_000.0),
    ("206K",   206_000.0),
    ("-21K",   -21_000.0),
    ("1.2B",   1_200_000_000.0),
    ("0.2%",   0.2),
    ("4.7%",   4.7),
    ("-0.1%",  -0.1),
    ("+0.3%",  0.3),
    ("58.3",   58.3),
    ("2.25",   2.25),
    ("1,234.5", 1234.5),
    ("−0.2%",  -0.2),     # unicode minus
])
def test_parse_economic_value_numeric(raw, expected):
    result = parse_economic_value(raw)
    assert result is not None
    assert abs(result - expected) < 0.001, f"parse({raw!r}) = {result}, expected {expected}"


@pytest.mark.parametrize("raw", ["N/A", "n/a", "", "  ", "-", "pending", "TBD", None])
def test_parse_economic_value_returns_none(raw):
    assert parse_economic_value(raw) is None


# ── SurpriseCalculator ────────────────────────────────────────────────────────

def test_nfp_beat_produces_bullish_small_surprise():
    """NFP 206K vs forecast 185K → +11.4% → SMALL BEAT."""
    event = make_event("US Non-Farm Employment Change", forecast="185K", actual="206K")
    result = SurpriseCalculator.compute(event, EventType.EMPLOYMENT)

    assert result is not None
    assert result.direction == SurpriseDirection.BEAT
    assert result.raw_surprise == pytest.approx(21_000.0)
    assert result.pct_surprise == pytest.approx(11.35, abs=0.1)
    assert result.surprise_class == SurpriseClass.SMALL


def test_cpi_large_beat():
    """CPI 0.5% vs forecast 0.2% → +150% pct surprise → EXTREME."""
    event = make_event("US CPI m/m", forecast="0.2%", actual="0.5%")
    result = SurpriseCalculator.compute(event, EventType.INFLATION)

    assert result is not None
    assert result.direction == SurpriseDirection.BEAT
    assert result.surprise_class == SurpriseClass.EXTREME


def test_pmi_miss():
    """PMI 52.0 vs forecast 55.0 → -5.45% → LARGE MISS."""
    event = make_event("US ISM Manufacturing PMI", forecast="55.0", actual="52.0")
    result = SurpriseCalculator.compute(event, EventType.PMI)

    assert result is not None
    assert result.direction == SurpriseDirection.MISS
    assert result.raw_surprise == pytest.approx(-3.0)


def test_inline_produces_none_class():
    """Result in line with forecast → IN_LINE."""
    event = make_event("US CPI m/m", forecast="0.2%", actual="0.2%")
    result = SurpriseCalculator.compute(event, EventType.INFLATION)

    assert result is not None
    assert result.direction == SurpriseDirection.IN_LINE
    assert result.surprise_class == SurpriseClass.NONE


def test_scheduled_event_returns_none():
    """Scheduled events (no actual) return None."""
    event = make_event(
        "US Non-Farm Employment Change",
        status=EventStatus.SCHEDULED,
        actual=None,
    )
    result = SurpriseCalculator.compute(event, EventType.EMPLOYMENT)
    assert result is None


def test_missing_actual_returns_none():
    event = make_event("US CPI m/m", actual=None)
    result = SurpriseCalculator.compute(event, EventType.INFLATION)
    assert result is None


def test_missing_forecast_returns_none():
    event = make_event("US CPI m/m", forecast=None)
    result = SurpriseCalculator.compute(event, EventType.INFLATION)
    assert result is None
