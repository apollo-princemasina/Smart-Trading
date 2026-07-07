from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel
from market_intel.models.enums import ImpactLevel, EventStatus, EventCategory


# ── Canonical event output ────────────────────────────────────────────────────

class MFIPEventOut(BaseModel):
    event_id:          str
    provider:          str
    provider_event_id: str
    title:             str
    currency:          str
    country:           str
    timestamp_utc:     Optional[datetime]
    is_all_day:        bool
    impact:            ImpactLevel
    is_high_impact:    bool
    is_speech:         bool
    category:          EventCategory
    forecast:          Optional[str]
    previous:          Optional[str]
    actual:            Optional[str]
    status:            EventStatus
    last_updated:      datetime
    metadata:          dict[str, Any]


# ── Calendar / event list responses ──────────────────────────────────────────

class CalendarResponse(BaseModel):
    week:       str
    is_stale:   bool
    fetched_at: datetime
    count:      int
    events:     list[MFIPEventOut]


class NextEventResponse(BaseModel):
    event:         Optional[MFIPEventOut]
    minutes_until: Optional[float]
    message:       str


# ── Health response ───────────────────────────────────────────────────────────

class JobHealthOut(BaseModel):
    job_id:          str
    status:          str            # "ok" | "degraded" | "down" | "initializing"
    poll_interval_s: int
    last_success:    Optional[datetime]
    last_failure:    Optional[datetime]
    next_run:        Optional[datetime]
    success_count:   int
    failure_count:   int
    retry_count:     int
    circuit_open:    bool
    avg_response_ms: Optional[float]
    last_response_ms: Optional[float]


class IntelligenceHealthOut(BaseModel):
    # Identity
    schema_version: str
    provider:       str

    # Connector lifecycle
    connector_status:  str            # "ok" | "degraded" | "initializing" | "down"
    scheduler_running: bool
    started_at:        Optional[datetime]
    uptime_s:          Optional[float]

    # Cache state
    cache_populated:             dict[str, bool]
    calendar_events_total:       int
    calendar_events_high_impact: int
    speeches_cached:             int
    news_items_cached:           int   # Always 0 until Phase 3

    # Per-job metrics
    jobs: dict[str, JobHealthOut]

    # Polling configuration
    calendar_poll_s:  int
    news_poll_s:      int
    sentiment_poll_s: int
    speeches_poll_s:  int
