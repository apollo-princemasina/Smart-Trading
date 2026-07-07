"""
ContextPayload and its component dataclasses.

ContextPayload is the single structured input that the Market Intelligence Agent
receives. It is assembled by the MarketContextCompiler from all available data sources.

`context_schema_version` must be bumped whenever fields are added or semantics change,
as the cache key includes this version — old entries automatically become stale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from market_intelligence_ai.models.enums import AnalysisType


@dataclass(frozen=True)
class EventTrigger:
    """Economic event data that triggered an analysis request."""
    event_id:           str
    title:              str
    currency:           str
    timestamp:          datetime
    importance:         str = "MEDIUM"    # HIGH | MEDIUM | LOW (from ImpactLevel)
    forecast:           Optional[str] = None
    actual:             Optional[str] = None
    previous:           Optional[str] = None
    surprise_class:     str = "NONE"      # NONE | SMALL | MEDIUM | LARGE | EXTREME
    surprise_direction: str = "IN_LINE"   # IN_LINE | BEAT | MISS
    economic_direction: str = "UNCERTAIN" # EIE deterministic direction


@dataclass(frozen=True)
class HeadlineTrigger:
    """Market headline that triggered an analysis request."""
    headline_id:         str
    headline:            str
    source:              str
    timestamp:           datetime
    affected_currencies: List[str] = field(default_factory=list)


@dataclass
class EIESnapshot:
    """
    Snapshot of the Economic Intelligence Engine state at the time of analysis.

    This is the deterministic context the AI receives. It is computed by the EIE
    and passed to the MarketContextCompiler — the AI layer never queries the EIE directly.
    """
    dominant_directions:  Dict[str, str]      = field(default_factory=dict)
    active_events:        List[Dict[str, Any]] = field(default_factory=list)
    upcoming_high_impact: List[Dict[str, Any]] = field(default_factory=list)
    execution_risk:       float = 0.0
    execution_readiness:  float = 0.0
    snapshot_at:          Optional[datetime] = None


@dataclass
class ContextPayload:
    """
    Fully assembled context package for one analysis request.

    This is what the Market Intelligence Agent receives as its user message.
    The MarketContextCompiler assembles this from all available data sources.
    The AI reasons from this context autonomously — no prompt selection, no routing.
    """
    context_schema_version: str = "context_v1"

    analysis_type:      AnalysisType = AnalysisType.EVENT
    primary_currency:   str = "USD"
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_session:    str = "UNKNOWN"  # ASIA | LONDON | NEW_YORK | OVERLAP | OFF_MARKET

    # Trigger data — at least one must be set
    event_trigger:    Optional[EventTrigger]    = None
    headline_trigger: Optional[HeadlineTrigger] = None

    # Economic intelligence context from EIE
    eie_snapshot: EIESnapshot = field(default_factory=EIESnapshot)

    # Live market context (price, regime, session) — only used for GENERAL type
    market_ctx: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.analysis_type != AnalysisType.GENERAL:
            if self.event_trigger is None and self.headline_trigger is None:
                raise ValueError("ContextPayload must have at least one trigger (event or headline).")
