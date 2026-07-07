"""Direction engine data models."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EconomicDirection(str, Enum):
    BULLISH   = "BULLISH"   # Event supports strength of the event's currency
    BEARISH   = "BEARISH"   # Event supports weakness of the event's currency
    NEUTRAL   = "NEUTRAL"   # No clear directional bias
    UNCERTAIN = "UNCERTAIN" # Insufficient data to determine direction


@dataclass(frozen=True)
class DirectionSignal:
    """Output of the DirectionRuleEngine for a single event."""
    direction:  EconomicDirection
    confidence: float   # 0.0 – 1.0
    rationale:  str     # Human-readable explanation
