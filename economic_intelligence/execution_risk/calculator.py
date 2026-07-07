"""
ExecutionRiskCalculator — computes Execution Risk Score and Execution Readiness Score.

Execution Risk (0–100): how dangerous it is to execute a trade right now.
  HIGH when:
    - HIGH-impact event is imminent (within 5–30 min)
    - Recent released event still carries strong influence
    - Multiple events are clustered in the next 30 min
    - Market is closed or it is a holiday

Execution Readiness (0–100): how confident and prepared to enter a trade right now.
  HIGH when:
    - No high-impact events in the next 60+ min
    - A strong surprise just released (post-release momentum window)
    - Clear directional signal with high confidence

The two scores are related but not strict inverses — a just-released EXTREME
surprise can produce moderate risk AND high readiness simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from economic_intelligence.surprise_engine.models import SurpriseClass
from economic_intelligence.direction_engine.models import EconomicDirection

if TYPE_CHECKING:
    from economic_intelligence.intelligence_models.models import EconomicIntelligenceReport


@dataclass(frozen=True)
class ExecutionContext:
    """Full snapshot of current execution conditions."""
    execution_risk:       float   # 0–100
    execution_readiness:  float   # 0–100
    risk_rationale:       str
    readiness_rationale:  str
    time_to_next_high_min: Optional[float]   # None = no HIGH event found
    active_event_count:   int
    upcoming_event_count: int
    is_market_open:       bool
    is_holiday:           bool


class ExecutionRiskCalculator:
    """
    Computes (risk_score, readiness_score, context) from the current intelligence snapshot.

    Parameters
    ----------
    upcoming_high : list of EconomicIntelligenceReport
        HIGH-impact events in the next EIE_RISK_LOOKAHEAD_MIN minutes (scheduled).
    active_events : list of EconomicIntelligenceReport
        Released events with remaining_influence > EIE_ACTIVE_THRESHOLD.
    is_market_open : bool
    is_holiday : bool
    """

    @staticmethod
    def compute(
        upcoming_high: "list[EconomicIntelligenceReport]",
        active_events: "list[EconomicIntelligenceReport]",
        is_market_open: bool,
        is_holiday: bool,
    ) -> ExecutionContext:

        # ── Holiday / market-closed override ─────────────────────────────────
        if is_holiday:
            return ExecutionContext(
                execution_risk=95.0, execution_readiness=5.0,
                risk_rationale="Market holiday — do not trade",
                readiness_rationale="Holiday session — no execution readiness",
                time_to_next_high_min=None,
                active_event_count=len(active_events),
                upcoming_event_count=len(upcoming_high),
                is_market_open=False,
                is_holiday=True,
            )

        if not is_market_open:
            return ExecutionContext(
                execution_risk=80.0, execution_readiness=20.0,
                risk_rationale="Forex market closed",
                readiness_rationale="Market closed — no execution readiness",
                time_to_next_high_min=_min_time_to_event(upcoming_high),
                active_event_count=len(active_events),
                upcoming_event_count=len(upcoming_high),
                is_market_open=False,
                is_holiday=False,
            )

        # ── Time risk (0–50 pts): proximity of next HIGH-impact event ─────────
        time_to_next = _min_time_to_event(upcoming_high)
        time_risk, time_note = _time_risk(time_to_next)

        # ── Influence risk (0–30 pts): how strongly recent events still affect ─
        max_influence = max((e.remaining_influence for e in active_events), default=0.0)
        influence_risk = max_influence * 0.30
        influence_note = (
            f"Active event influence at {max_influence:.0f}%" if active_events else
            "No active events"
        )

        # ── Cluster risk (0–20 pts): events bunched together ─────────────────
        events_in_30min = sum(
            1 for e in upcoming_high
            if e.time_to_event is not None and e.time_to_event <= 30
        )
        cluster_risk = min(20.0, events_in_30min * 7.0)
        cluster_note = (
            f"{events_in_30min} HIGH event(s) within 30 min" if events_in_30min else
            "No event clustering"
        )

        execution_risk = min(100.0, time_risk + influence_risk + cluster_risk)

        # ── Readiness ─────────────────────────────────────────────────────────
        # Post-release boost: strong signal released recently (within last 30 min)
        recent_strong = [
            e for e in active_events
            if (
                e.event_age_hours is not None
                and e.event_age_hours <= 0.5
                and e.surprise_class in (SurpriseClass.LARGE, SurpriseClass.EXTREME)
                and e.economic_direction in (EconomicDirection.BULLISH, EconomicDirection.BEARISH)
            )
        ]
        post_release_boost = 20.0 if recent_strong else 0.0
        raw_readiness = max(0.0, 100.0 - execution_risk + post_release_boost)
        execution_readiness = min(100.0, raw_readiness)

        readiness_note = (
            f"Post-release momentum window — {recent_strong[0].event_title} released" if recent_strong else
            f"Risk-adjusted readiness ({100.0 - execution_risk:.0f}%)"
        )

        return ExecutionContext(
            execution_risk=round(execution_risk, 1),
            execution_readiness=round(execution_readiness, 1),
            risk_rationale=f"{time_note}; {influence_note}; {cluster_note}",
            readiness_rationale=readiness_note,
            time_to_next_high_min=time_to_next,
            active_event_count=len(active_events),
            upcoming_event_count=len(upcoming_high),
            is_market_open=True,
            is_holiday=False,
        )


def _min_time_to_event(events: "list[EconomicIntelligenceReport]") -> Optional[float]:
    times = [e.time_to_event for e in events if e.time_to_event is not None]
    return min(times) if times else None


def _time_risk(time_to_next: Optional[float]) -> tuple[float, str]:
    if time_to_next is None:
        return 0.0, "No HIGH-impact events in lookahead window"
    if time_to_next < 5:
        return 50.0, f"HIGH event in {time_to_next:.0f} min — DO NOT TRADE"
    if time_to_next < 15:
        return 40.0, f"HIGH event in {time_to_next:.0f} min — extreme caution"
    if time_to_next < 30:
        return 30.0, f"HIGH event in {time_to_next:.0f} min — caution"
    if time_to_next < 60:
        return 18.0, f"HIGH event in {time_to_next:.0f} min — moderate risk"
    if time_to_next < 120:
        return 8.0, f"HIGH event in {time_to_next:.0f} min — low risk"
    return 0.0, f"Next HIGH event in {time_to_next:.0f} min — minimal risk"
