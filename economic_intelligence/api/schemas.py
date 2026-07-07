"""
Pydantic schemas for EIE API responses.

These schemas are the public contract between the EIE and any consumer.
Internal engine types (SurpriseResult, DirectionRule, etc.) are never exposed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.direction_engine.models import EconomicDirection
from economic_intelligence.surprise_engine.models import SurpriseClass, SurpriseDirection
from market_intel.models.enums import ImpactLevel


class EconomicIntelligenceOut(BaseModel):
    """Single event's full economic intelligence — the primary EIE output."""

    # Identity
    report_id:    str
    event_id:     str
    generated_at: datetime

    # Event fields
    event_title:   str
    currency:      str
    country:       str
    timestamp_utc: Optional[datetime]
    is_released:   bool
    importance:    ImpactLevel

    # Classification
    event_type: EventType

    # Impact
    impact_score: float = Field(ge=0, le=100)

    # Surprise
    surprise:           Optional[float]
    pct_surprise:       Optional[float]
    surprise_class:     SurpriseClass
    surprise_direction: SurpriseDirection

    # Direction
    economic_direction:   EconomicDirection
    direction_confidence: float = Field(ge=0, le=1)
    direction_rationale:  str

    # Decay
    remaining_influence: float = Field(ge=0, le=100)
    event_age_hours:     Optional[float]

    # Timing
    time_to_event: Optional[float]

    # Execution
    execution_risk:      float = Field(ge=0, le=100)
    execution_readiness: float = Field(ge=0, le=100)

    # Meta
    confidence:   float = Field(ge=0, le=1)
    last_updated: datetime

    class Config:
        from_attributes = True


class ExecutionContextOut(BaseModel):
    """Current market execution context."""
    execution_risk:        float = Field(ge=0, le=100)
    execution_readiness:   float = Field(ge=0, le=100)
    risk_rationale:        str
    readiness_rationale:   str
    time_to_next_high_min: Optional[float]
    active_event_count:    int
    upcoming_event_count:  int
    is_market_open:        bool
    is_holiday:            bool
    generated_at:          datetime


class IntelligenceContextResponse(BaseModel):
    """Full context response — active events + upcoming events + execution context."""
    context:         ExecutionContextOut
    active_events:   list[EconomicIntelligenceOut]
    upcoming_events: list[EconomicIntelligenceOut]
    total_active:    int
    total_upcoming:  int
    generated_at:    datetime


class ActiveEventsResponse(BaseModel):
    events:       list[EconomicIntelligenceOut]
    count:        int
    generated_at: datetime


class UpcomingEventsResponse(BaseModel):
    events:       list[EconomicIntelligenceOut]
    count:        int
    hours_ahead:  float
    generated_at: datetime


class CurrencyDirectionOut(BaseModel):
    """Economic direction summary for a single currency."""
    currency:             str
    dominant_direction:   EconomicDirection
    avg_confidence:       float
    avg_impact_score:     float
    avg_remaining_influence: float
    active_event_count:   int
    bullish_count:        int
    bearish_count:        int
    neutral_count:        int


class EconomicSummaryResponse(BaseModel):
    """Summary of economic direction by currency."""
    currencies:     list[CurrencyDirectionOut]
    generated_at:   datetime
    total_reports:  int
