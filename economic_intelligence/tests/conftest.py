"""Shared test fixtures for the EIE test suite."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from market_intel.models.enums import ImpactLevel, EventStatus, EventCategory, Provider
from market_intel.models.event import MFIPEvent


def make_event(
    title: str,
    currency: str = "USD",
    impact: ImpactLevel = ImpactLevel.HIGH,
    status: EventStatus = EventStatus.RELEASED,
    forecast: str | None = "185K",
    actual: str | None = "206K",
    timestamp_offset_hours: float = -1.0,  # negative = past
) -> MFIPEvent:
    """Factory for MFIPEvent test instances."""
    now = datetime.now(timezone.utc)
    ts = now + timedelta(hours=timestamp_offset_hours)

    return MFIPEvent(
        event_id="test_" + title[:8].lower().replace(" ", "_"),
        provider=Provider.FOREX_FACTORY,
        provider_event_id="prov_" + title[:6].lower().replace(" ", "_"),
        title=title,
        currency=currency,
        country="US" if currency == "USD" else "EU",
        timestamp_utc=ts,
        is_all_day=False,
        impact=impact,
        is_high_impact=impact == ImpactLevel.HIGH,
        is_speech="speaks" in title.lower() or "speech" in title.lower(),
        category=EventCategory.EMPLOYMENT,
        forecast=forecast,
        previous="177K",
        actual=actual,
        status=status,
        last_updated=now,
    )
