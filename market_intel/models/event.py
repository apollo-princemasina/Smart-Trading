from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
from .enums import ImpactLevel, EventStatus, EventCategory


class MFIPEvent(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────────
    # Global unique ID: SHA-256(provider + currency + timestamp_utc + title).
    # Stable across re-fetches; the same real-world event always gets the same ID.
    event_id: str

    # Which data provider produced this event.
    provider: str                    # Provider enum value as string

    # The provider's own internal identifier for this event.
    # Used for deduplication within a single provider's data stream.
    provider_event_id: str

    # ── Descriptive ──────────────────────────────────────────────────────────
    title: str
    currency: str                    # ISO 4217: EUR, USD, GBP, JPY, CHF, CAD, AUD, NZD, …
    country: str                     # ISO 3166-1 alpha-2: US, EU, GB, JP, CH, CA, AU, NZ, …

    # ── Timing ───────────────────────────────────────────────────────────────
    timestamp_utc: Optional[datetime] = None   # None for all-day events (holidays, etc.)
    is_all_day: bool = False

    # ── Impact ───────────────────────────────────────────────────────────────
    impact: ImpactLevel
    is_high_impact: bool             # Convenience flag: impact == HIGH

    # ── Classification ───────────────────────────────────────────────────────
    is_speech: bool                  # Central bank speech, testimony, press conference
    category: EventCategory

    # ── Economic data ─────────────────────────────────────────────────────────
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None     # None = not yet released

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    status: EventStatus

    # ── Housekeeping ──────────────────────────────────────────────────────────
    # UTC timestamp of the last time any field on this event changed.
    last_updated: datetime

    # Provider-specific raw fields, source URLs, revision markers, etc.
    # Consumers should not depend on this — it is for debugging and audit only.
    metadata: dict[str, Any] = Field(default_factory=dict)
