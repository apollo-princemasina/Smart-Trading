import hashlib
from datetime import datetime
from typing import Optional


def build_provider_event_id(event_name: str, currency: str, timestamp_utc: Optional[datetime]) -> str:
    """Provider-scoped ID — unique within Forex Factory's data stream."""
    time_part = timestamp_utc.isoformat() if timestamp_utc else "allday"
    key = f"{event_name}|{currency}|{time_part}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_event_id(provider: str, event_name: str, currency: str, timestamp_utc: Optional[datetime]) -> str:
    """
    Global unique ID — stable across providers and re-fetches.

    Including provider ensures that the same event (e.g. US CPI) sourced from
    both Forex Factory and FXStreet gets two distinct event_ids, allowing the
    deduplication layer to merge or prefer one provider over the other.
    """
    time_part = timestamp_utc.isoformat() if timestamp_utc else "allday"
    key = f"{provider}|{event_name}|{currency}|{time_part}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
