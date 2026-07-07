from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Query, HTTPException, Request
from market_intel.models.enums import EventStatus

from ...cache.memory_cache import connector_cache, CacheNotReadyError
from ..schemas import CalendarResponse, MFIPEventOut, NextEventResponse

router = APIRouter(prefix="/events", tags=["Intelligence — Events"])


def _to_response(week: str, cache, events) -> CalendarResponse:
    return CalendarResponse(
        week=week,
        is_stale=cache.is_stale,
        fetched_at=cache.fetched_at,
        count=len(events),
        events=[MFIPEventOut(**e.model_dump()) for e in events],
    )


@router.get("/today", response_model=CalendarResponse, summary="All events scheduled for today (UTC)")
async def get_today(
    currency: str | None = Query(None, description="Filter by ISO currency code, e.g. USD"),
):
    try:
        cache = await connector_cache.get_calendar("thisweek")
    except CacheNotReadyError:
        raise HTTPException(503, detail="Calendar cache not yet populated — try again shortly")

    today = datetime.now(timezone.utc).date()
    events = [
        e for e in cache.events
        if e.timestamp_utc and e.timestamp_utc.date() == today
    ]
    if currency:
        events = [e for e in events if e.currency.upper() == currency.upper()]

    events.sort(key=lambda e: e.timestamp_utc)
    return _to_response("thisweek", cache, events)


@router.get(
    "/high-impact",
    response_model=CalendarResponse,
    summary="All HIGH-impact events for the specified week",
)
async def get_high_impact(
    week: str = Query("thisweek", pattern="^(thisweek|nextweek|lastweek)$"),
    currency: str | None = Query(None),
):
    try:
        cache = await connector_cache.get_calendar(week)
    except CacheNotReadyError:
        raise HTTPException(503, detail="Calendar cache not yet populated")

    events = [e for e in cache.events if e.is_high_impact]
    if currency:
        events = [e for e in events if e.currency.upper() == currency.upper()]

    events.sort(key=lambda e: e.timestamp_utc or datetime.max.replace(tzinfo=timezone.utc))
    return _to_response(week, cache, events)


@router.get(
    "/next",
    response_model=NextEventResponse,
    summary="The single next upcoming event from now (searches thisweek then nextweek)",
)
async def get_next_event(
    high_impact_only: bool = Query(False, description="Restrict to HIGH-impact events only"),
):
    now = datetime.now(timezone.utc)

    for week in ("thisweek", "nextweek"):
        try:
            cache = await connector_cache.get_calendar(week)
        except CacheNotReadyError:
            continue

        candidates = [
            e for e in cache.events
            if e.timestamp_utc
            and e.timestamp_utc > now
            and e.status == EventStatus.SCHEDULED
            and (not high_impact_only or e.is_high_impact)
        ]
        if candidates:
            nxt = min(candidates, key=lambda e: e.timestamp_utc)
            minutes_until = round((nxt.timestamp_utc - now).total_seconds() / 60, 1)
            return NextEventResponse(
                event=MFIPEventOut(**nxt.model_dump()),
                minutes_until=minutes_until,
                message=f"Next event in {minutes_until:.0f} min",
            )

    return NextEventResponse(event=None, minutes_until=None, message="No upcoming events found")
