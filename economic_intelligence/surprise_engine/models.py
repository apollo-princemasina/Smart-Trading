"""Surprise engine data models."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SurpriseClass(str, Enum):
    NONE    = "NONE"     # actual == forecast (within threshold)
    SMALL   = "SMALL"    # slight beat or miss
    MEDIUM  = "MEDIUM"   # notable beat or miss
    LARGE   = "LARGE"    # significant beat or miss
    EXTREME = "EXTREME"  # shock — very far from consensus


class SurpriseDirection(str, Enum):
    BEAT    = "BEAT"     # actual > forecast (positive surprise)
    MISS    = "MISS"     # actual < forecast (negative surprise)
    IN_LINE = "IN_LINE"  # essentially in line with forecast


@dataclass(frozen=True)
class SurpriseResult:
    """Fully computed surprise for a single event."""

    # Raw values parsed to float
    actual_value:   float
    forecast_value: float

    # Absolute difference (actual - forecast), in original units
    raw_surprise: float

    # (actual - forecast) / |forecast| * 100 — None when forecast ≈ 0
    pct_surprise: Optional[float]

    # Direction: BEAT / MISS / IN_LINE
    direction: SurpriseDirection

    # Magnitude classification
    surprise_class: SurpriseClass

    @property
    def is_significant(self) -> bool:
        return self.surprise_class in (SurpriseClass.MEDIUM, SurpriseClass.LARGE, SurpriseClass.EXTREME)
