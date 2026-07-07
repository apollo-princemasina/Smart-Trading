"""
DecayCalculator — computes the remaining influence of a released event.

Remaining Influence = base_score * 0.5^(hours_elapsed / half_life)
Clamped to [min_influence, base_score].

For scheduled events (not yet released), remaining_influence = 0.0.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from market_intel.models.enums import EventStatus
from market_intel.models.event import MFIPEvent
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.decay_engine.curves import get_decay_profile


class DecayCalculator:
    """Computes remaining influence (0–100) and event age for a released event."""

    @staticmethod
    def compute(
        event: MFIPEvent,
        event_type: EventType,
        base_impact_score: float,
        now: Optional[datetime] = None,
    ) -> tuple[float, Optional[float]]:
        """
        Returns (remaining_influence, event_age_hours).

        remaining_influence: 0.0 – 100.0
        event_age_hours:     hours since release, or None if not released / no timestamp
        """
        if event.status == EventStatus.SCHEDULED:
            return 0.0, None

        if event.timestamp_utc is None:
            return 0.0, None

        now = now or datetime.now(timezone.utc)
        elapsed_seconds = (now - event.timestamp_utc).total_seconds()

        if elapsed_seconds < 0:
            # Event is in the future — not yet released despite status
            return 0.0, None

        hours_elapsed = elapsed_seconds / 3600.0
        profile = get_decay_profile(event_type)

        # Exponential half-life decay: influence = base * 0.5^(t / t½)
        remaining = base_impact_score * math.pow(0.5, hours_elapsed / profile.half_life_hours)
        remaining = max(profile.min_influence, min(base_impact_score, remaining))

        return round(remaining, 2), round(hours_elapsed, 3)
