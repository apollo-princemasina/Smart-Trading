"""GET /intelligence/active-events and /intelligence/upcoming-events endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query

from economic_intelligence.intelligence_cache.cache import intelligence_cache
from economic_intelligence.api.schemas import (
    ActiveEventsResponse,
    UpcomingEventsResponse,
    EconomicIntelligenceOut,
)

router = APIRouter()


def _to_out(r) -> EconomicIntelligenceOut:
    return EconomicIntelligenceOut(**r.__dict__)


@router.get("/active-events", response_model=ActiveEventsResponse)
async def get_active_events(
    currency: Optional[str] = Query(None, description="Filter by ISO 4217 currency code"),
):
    """
    Released events that still carry remaining influence above the active threshold.

    These are the events currently driving market sentiment.
    """
    events = await intelligence_cache.get_active()

    if currency:
        events = [e for e in events if e.currency == currency.upper()]

    events.sort(key=lambda r: r.remaining_influence, reverse=True)

    return ActiveEventsResponse(
        events=[_to_out(r) for r in events],
        count=len(events),
        generated_at=datetime.now(timezone.utc),
    )


@router.get("/upcoming-events", response_model=UpcomingEventsResponse)
async def get_upcoming_events(
    hours_ahead: float = Query(24.0, ge=0.5, le=168.0, description="Look-ahead window in hours (max 168)"),
    currency: Optional[str] = Query(None, description="Filter by ISO 4217 currency code"),
    high_impact_only: bool = Query(False, description="Return only HIGH-impact events"),
):
    """
    Scheduled economic events within the next `hours_ahead` hours.

    Sorted by timestamp (earliest first).
    """
    from market_intel.models.enums import ImpactLevel

    events = await intelligence_cache.get_upcoming(limit_hours=hours_ahead)

    if currency:
        events = [e for e in events if e.currency == currency.upper()]

    if high_impact_only:
        events = [e for e in events if e.importance == ImpactLevel.HIGH]

    return UpcomingEventsResponse(
        events=[_to_out(r) for r in events],
        count=len(events),
        hours_ahead=hours_ahead,
        generated_at=datetime.now(timezone.utc),
    )
