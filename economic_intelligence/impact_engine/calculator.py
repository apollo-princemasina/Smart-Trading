"""
ImpactCalculator — converts event importance → Economic Impact Score (0–100).

Formula:
  base  = BASE_IMPACT_SCORES[impact_level]
  score = base * type_multiplier
  if released and surprise available:
      score *= SURPRISE_AMPLIFIERS[surprise.surprise_class]
  score = clamp(score, 0, 100)
"""
from __future__ import annotations

from typing import Optional

from market_intel.models.enums import ImpactLevel
from market_intel.models.event import MFIPEvent
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.impact_engine.weights import (
    BASE_IMPACT_SCORES,
    EVENT_TYPE_MULTIPLIERS,
    SURPRISE_AMPLIFIERS,
    _DEFAULT_TYPE_MULTIPLIER,
)
from economic_intelligence.surprise_engine.models import SurpriseResult


class ImpactCalculator:
    """Computes the Economic Impact Score (0–100) for a single event."""

    @staticmethod
    def compute(
        event: MFIPEvent,
        event_type: EventType,
        surprise: Optional[SurpriseResult] = None,
    ) -> float:
        base = BASE_IMPACT_SCORES.get(event.impact, BASE_IMPACT_SCORES[ImpactLevel.LOW])
        type_mult = EVENT_TYPE_MULTIPLIERS.get(event_type, _DEFAULT_TYPE_MULTIPLIER)

        score = base * type_mult

        if surprise is not None:
            amplifier = SURPRISE_AMPLIFIERS.get(surprise.surprise_class, 1.0)
            score *= amplifier

        return min(100.0, max(0.0, score))
