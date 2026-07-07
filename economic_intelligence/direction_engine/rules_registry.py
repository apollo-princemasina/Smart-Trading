"""
Centralized direction rules registry.

Each rule defines whether a HIGHER actual reading (relative to forecast)
is bullish or bearish for the event's currency.

  higher_is_bullish = True  → positive surprise (BEAT) → BULLISH
  higher_is_bullish = False → positive surprise (BEAT) → BEARISH
                               (e.g., unemployment: more jobless = bad = bearish)

All rules are applied identically regardless of currency — the direction
is always from the perspective of the event's own currency.
"""
from __future__ import annotations

from dataclasses import dataclass
from economic_intelligence.event_classifier.event_types import EventType


@dataclass(frozen=True)
class DirectionRule:
    event_type:        EventType
    higher_is_bullish: bool     # True → BEAT = BULLISH; False → BEAT = BEARISH
    confidence:        float    # Base confidence (0.0 – 1.0)
    rationale:         str      # Explanation shown in API responses


# ── Registry (keyed by EventType) ─────────────────────────────────────────────
DIRECTION_RULES: dict[EventType, DirectionRule] = {

    EventType.EMPLOYMENT: DirectionRule(
        event_type=EventType.EMPLOYMENT,
        higher_is_bullish=True,
        confidence=0.80,
        rationale="Higher employment → strong labor market → bullish currency",
    ),

    EventType.UNEMPLOYMENT: DirectionRule(
        event_type=EventType.UNEMPLOYMENT,
        higher_is_bullish=False,   # Higher unemployment = bad = bearish
        confidence=0.78,
        rationale="Higher unemployment → weak labor market → bearish currency",
    ),

    EventType.JOBLESS_CLAIMS: DirectionRule(
        event_type=EventType.JOBLESS_CLAIMS,
        higher_is_bullish=False,   # More claims = more layoffs = bearish
        confidence=0.72,
        rationale="Higher jobless claims → rising layoffs → bearish currency",
    ),

    EventType.WAGES: DirectionRule(
        event_type=EventType.WAGES,
        higher_is_bullish=True,
        confidence=0.70,
        rationale="Higher wages → inflation expectations → hawkish CB → bullish currency",
    ),

    EventType.INFLATION: DirectionRule(
        event_type=EventType.INFLATION,
        higher_is_bullish=True,
        confidence=0.68,
        rationale="Higher inflation → hawkish central bank expectations → bullish currency",
    ),

    EventType.INTEREST_RATE: DirectionRule(
        event_type=EventType.INTEREST_RATE,
        higher_is_bullish=True,
        confidence=0.85,
        rationale="Higher rates → capital inflows → bullish currency",
    ),

    EventType.GDP: DirectionRule(
        event_type=EventType.GDP,
        higher_is_bullish=True,
        confidence=0.75,
        rationale="Higher GDP → strong economic output → bullish currency",
    ),

    EventType.RETAIL_SALES: DirectionRule(
        event_type=EventType.RETAIL_SALES,
        higher_is_bullish=True,
        confidence=0.65,
        rationale="Higher retail sales → consumer strength → bullish currency",
    ),

    EventType.CONSUMER_CONFIDENCE: DirectionRule(
        event_type=EventType.CONSUMER_CONFIDENCE,
        higher_is_bullish=True,
        confidence=0.60,
        rationale="Higher confidence → economic optimism → bullish currency",
    ),

    EventType.PMI: DirectionRule(
        event_type=EventType.PMI,
        higher_is_bullish=True,
        confidence=0.65,
        rationale="Higher PMI → economic expansion → bullish currency",
    ),

    EventType.MANUFACTURING: DirectionRule(
        event_type=EventType.MANUFACTURING,
        higher_is_bullish=True,
        confidence=0.62,
        rationale="Higher manufacturing output → industrial strength → bullish currency",
    ),

    EventType.INDUSTRIAL: DirectionRule(
        event_type=EventType.INDUSTRIAL,
        higher_is_bullish=True,
        confidence=0.60,
        rationale="Higher industrial production → economic activity → bullish currency",
    ),

    EventType.TRADE_BALANCE: DirectionRule(
        event_type=EventType.TRADE_BALANCE,
        higher_is_bullish=True,
        confidence=0.55,
        rationale="Better trade balance → stronger demand for currency → bullish",
    ),

    EventType.HOUSING: DirectionRule(
        event_type=EventType.HOUSING,
        higher_is_bullish=True,
        confidence=0.55,
        rationale="Higher housing activity → economic confidence → bullish currency",
    ),

    # Central bank speech: ambiguous without full text — low confidence
    EventType.CENTRAL_BANK_SPEECH: DirectionRule(
        event_type=EventType.CENTRAL_BANK_SPEECH,
        higher_is_bullish=True,
        confidence=0.40,
        rationale="Central bank speech — direction uncertain without text analysis",
    ),
}
