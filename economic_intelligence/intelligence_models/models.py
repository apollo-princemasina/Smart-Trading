"""
EconomicIntelligenceReport — the canonical output object of the EIE pipeline.

This is the single contract between the EIE and all consumers (API, frontend,
future strategy engine). No engine-internal types (SurpriseResult, DirectionRule,
DecayProfile) ever leave this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from market_intel.models.enums import ImpactLevel
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.direction_engine.models import EconomicDirection
from economic_intelligence.surprise_engine.models import SurpriseClass, SurpriseDirection


@dataclass
class EconomicIntelligenceReport:
    # ── Identity ──────────────────────────────────────────────────────────────
    report_id:       str           # SHA-256 of (event_id + generated_at_iso)
    event_id:        str           # Stable event ID from MFIPEvent
    generated_at:    datetime      # UTC timestamp of this computation

    # ── Event fields (denormalised from MFIPEvent) ────────────────────────────
    event_title:     str
    currency:        str           # ISO 4217
    country:         str           # ISO 3166-1 alpha-2
    timestamp_utc:   Optional[datetime]
    is_released:     bool
    importance:      ImpactLevel

    # ── Classification ────────────────────────────────────────────────────────
    event_type:      EventType

    # ── Impact ────────────────────────────────────────────────────────────────
    impact_score:    float         # 0–100

    # ── Surprise (None when not yet released or no forecast) ─────────────────
    surprise:           Optional[float]         # raw_surprise (actual - forecast)
    pct_surprise:       Optional[float]         # percentage surprise
    surprise_class:     SurpriseClass           # NONE / SMALL / MEDIUM / LARGE / EXTREME
    surprise_direction: SurpriseDirection       # BEAT / MISS / IN_LINE

    # ── Direction ─────────────────────────────────────────────────────────────
    economic_direction:   EconomicDirection     # BULLISH / BEARISH / NEUTRAL / UNCERTAIN
    direction_confidence: float                 # 0.0 – 1.0
    direction_rationale:  str

    # ── Decay ─────────────────────────────────────────────────────────────────
    remaining_influence: float                 # 0–100, decays after release
    event_age_hours:     Optional[float]       # Hours since release

    # ── Timing ────────────────────────────────────────────────────────────────
    time_to_event:       Optional[float]       # Minutes until event (negative = past)

    # ── Execution ─────────────────────────────────────────────────────────────
    execution_risk:      float                 # 0–100 (computed from context, not per-event)
    execution_readiness: float                 # 0–100

    # ── Meta ──────────────────────────────────────────────────────────────────
    confidence:          float                 # Overall pipeline confidence 0–1
    last_updated:        datetime              # UTC timestamp of last recompute
