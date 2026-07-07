"""
DirectionRuleEngine — resolves economic direction from event + surprise.

Decision matrix:
  ┌─────────────────┬─────────────────┬──────────────┐
  │  Event Status   │  Surprise Dir   │  Output      │
  ├─────────────────┼─────────────────┼──────────────┤
  │  SCHEDULED      │  N/A (no data)  │  UNCERTAIN   │
  │  RELEASED, BEAT │  higher_bullish │  BULLISH     │
  │  RELEASED, BEAT │  !higher_bullish│  BEARISH     │
  │  RELEASED, MISS │  higher_bullish │  BEARISH     │
  │  RELEASED, MISS │  !higher_bullish│  BULLISH     │
  │  RELEASED, LINE │  any            │  NEUTRAL     │
  │  No rule found  │  any            │  UNCERTAIN   │
  └─────────────────┴─────────────────┴──────────────┘
"""
from __future__ import annotations

from typing import Optional

from market_intel.models.enums import EventStatus
from market_intel.models.event import MFIPEvent
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.direction_engine.models import DirectionSignal, EconomicDirection
from economic_intelligence.direction_engine.rules_registry import DIRECTION_RULES
from economic_intelligence.surprise_engine.models import SurpriseDirection, SurpriseResult

# Types that produce UNCERTAIN direction even when surprise is available
_UNCERTAIN_TYPES = frozenset({
    EventType.POLITICAL,
    EventType.HOLIDAY,
    EventType.OIL_INVENTORY,
    EventType.UNKNOWN,
})

_UNCERTAIN_SIGNAL = DirectionSignal(
    direction=EconomicDirection.UNCERTAIN,
    confidence=0.0,
    rationale="Insufficient data to determine economic direction",
)

_NEUTRAL_SIGNAL = DirectionSignal(
    direction=EconomicDirection.NEUTRAL,
    confidence=0.50,
    rationale="Actual in line with forecast — no directional bias",
)


class DirectionRuleEngine:
    """
    Fully deterministic direction resolver.

    Returns a DirectionSignal with:
    - direction: BULLISH / BEARISH / NEUTRAL / UNCERTAIN
    - confidence: 0.0 – 1.0
    - rationale: plain English explanation
    """

    @staticmethod
    def resolve(
        event: MFIPEvent,
        event_type: EventType,
        surprise: Optional[SurpriseResult],
    ) -> DirectionSignal:

        if event_type in _UNCERTAIN_TYPES:
            return _UNCERTAIN_SIGNAL

        if event.status == EventStatus.SCHEDULED or surprise is None:
            return DirectionSignal(
                direction=EconomicDirection.UNCERTAIN,
                confidence=0.0,
                rationale=f"{event_type.value} event pending — direction unknown until released",
            )

        if surprise.direction == SurpriseDirection.IN_LINE:
            rule = DIRECTION_RULES.get(event_type)
            return DirectionSignal(
                direction=EconomicDirection.NEUTRAL,
                confidence=rule.confidence * 0.5 if rule else 0.40,
                rationale=f"Actual in line with forecast — no significant {event_type.value} surprise",
            )

        rule = DIRECTION_RULES.get(event_type)
        if rule is None:
            return _UNCERTAIN_SIGNAL

        beat = surprise.direction == SurpriseDirection.BEAT

        if beat == rule.higher_is_bullish:
            direction = EconomicDirection.BULLISH
        else:
            direction = EconomicDirection.BEARISH

        # Scale confidence by surprise magnitude
        magnitude_boost = {
            "NONE":    0.80,
            "SMALL":   0.90,
            "MEDIUM":  1.00,
            "LARGE":   1.10,
            "EXTREME": 1.15,
        }
        boost = magnitude_boost.get(surprise.surprise_class.value, 1.0)
        confidence = min(1.0, rule.confidence * boost)

        direction_word = "beat" if beat else "miss"
        rationale = (
            f"{event_type.value} {direction_word} — "
            f"{surprise.surprise_class.value} surprise "
            f"({'+' if surprise.raw_surprise >= 0 else ''}"
            f"{surprise.raw_surprise:.4g}) → "
            f"{rule.rationale}"
        )

        return DirectionSignal(
            direction=direction,
            confidence=confidence,
            rationale=rationale,
        )
