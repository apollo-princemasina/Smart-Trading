"""
SurpriseCalculator — computes the economic surprise for a released event.

All thresholds are configurable per EventType, with sensible defaults.
The surprise is always computed in the same units as the forecast/actual fields.
"""
from __future__ import annotations

from typing import Optional

from market_intel.models.event import MFIPEvent
from market_intel.models.enums import EventStatus
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.surprise_engine.models import (
    SurpriseClass,
    SurpriseDirection,
    SurpriseResult,
)
from economic_intelligence.surprise_engine.value_parser import parse_economic_value
from economic_intelligence.utils.config import eie_config

# ── Surprise classification thresholds ────────────────────────────────────────
# Keys are absolute |pct_surprise| boundaries (%).
# For each EventType, thresholds define the edges of SMALL / MEDIUM / LARGE / EXTREME.
# If |pct_surprise| < small → NONE; < medium → SMALL; < large → MEDIUM; < extreme → LARGE; else EXTREME.
# "IN_LINE" tolerance: |raw_surprise / |forecast|| < in_line_pct/100

_DEFAULT_THRESHOLDS = {
    "in_line": 2.0,   # ±2% → IN_LINE
    "small":   8.0,   # ±8% → SMALL
    "medium":  20.0,  # ±20% → MEDIUM
    "large":   40.0,  # ±40% → LARGE
    # ≥40% → EXTREME
}

_TYPE_THRESHOLDS: dict[EventType, dict[str, float]] = {
    EventType.EMPLOYMENT:          {"in_line": 3.0,  "small": 10.0, "medium": 25.0, "large": 50.0},
    EventType.UNEMPLOYMENT:        {"in_line": 1.0,  "small": 3.0,  "medium": 7.0,  "large": 15.0},
    EventType.JOBLESS_CLAIMS:      {"in_line": 2.0,  "small": 5.0,  "medium": 12.0, "large": 25.0},
    EventType.WAGES:               {"in_line": 3.0,  "small": 8.0,  "medium": 18.0, "large": 35.0},
    EventType.INFLATION:           {"in_line": 5.0,  "small": 15.0, "medium": 35.0, "large": 70.0},
    EventType.INTEREST_RATE:       {"in_line": 0.0,  "small": 5.0,  "medium": 15.0, "large": 30.0},
    EventType.GDP:                 {"in_line": 5.0,  "small": 15.0, "medium": 30.0, "large": 60.0},
    EventType.PMI:                 {"in_line": 0.5,  "small": 1.5,  "medium": 3.5,  "large": 7.0},
    EventType.RETAIL_SALES:        {"in_line": 3.0,  "small": 10.0, "medium": 25.0, "large": 50.0},
    EventType.CONSUMER_CONFIDENCE: {"in_line": 2.0,  "small": 5.0,  "medium": 12.0, "large": 25.0},
    EventType.TRADE_BALANCE:       {"in_line": 5.0,  "small": 15.0, "medium": 35.0, "large": 70.0},
}


def _classify_surprise(abs_pct: float, thresholds: dict[str, float]) -> SurpriseClass:
    if abs_pct <= thresholds["in_line"]:
        return SurpriseClass.NONE
    if abs_pct <= thresholds["small"]:
        return SurpriseClass.SMALL
    if abs_pct <= thresholds["medium"]:
        return SurpriseClass.MEDIUM
    if abs_pct <= thresholds["large"]:
        return SurpriseClass.LARGE
    return SurpriseClass.EXTREME


class SurpriseCalculator:
    """
    Computes the economic surprise for a released event.

    Returns None when:
    - actual is absent or non-numeric
    - forecast is absent or non-numeric
    - event status is SCHEDULED (not yet released)
    """

    @staticmethod
    def compute(event: MFIPEvent, event_type: EventType) -> Optional[SurpriseResult]:
        if event.status == EventStatus.SCHEDULED:
            return None

        actual_f   = parse_economic_value(event.actual)
        forecast_f = parse_economic_value(event.forecast)

        if actual_f is None or forecast_f is None:
            return None

        raw_surprise = actual_f - forecast_f

        # Percentage surprise
        abs_forecast = abs(forecast_f)
        if abs_forecast < eie_config.EIE_MIN_FORECAST_ABS:
            pct_surprise = None
            abs_pct = abs(raw_surprise) * 100.0  # fall back to raw magnitude in %
        else:
            pct_surprise = (raw_surprise / abs_forecast) * 100.0
            abs_pct = abs(pct_surprise)

        # Direction
        if abs_pct <= _TYPE_THRESHOLDS.get(event_type, _DEFAULT_THRESHOLDS)["in_line"]:
            direction = SurpriseDirection.IN_LINE
        elif raw_surprise > 0:
            direction = SurpriseDirection.BEAT
        else:
            direction = SurpriseDirection.MISS

        # Magnitude
        thresholds = _TYPE_THRESHOLDS.get(event_type, _DEFAULT_THRESHOLDS)
        surprise_class = _classify_surprise(abs_pct, thresholds)

        return SurpriseResult(
            actual_value=actual_f,
            forecast_value=forecast_f,
            raw_surprise=raw_surprise,
            pct_surprise=pct_surprise,
            direction=direction,
            surprise_class=surprise_class,
        )
