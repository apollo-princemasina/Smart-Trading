from fastapi import APIRouter, Query, HTTPException
from market_intel.models.enums import ImpactLevel
from ...cache.memory_cache import connector_cache, CacheNotReadyError
from ..schemas import CalendarResponse, MFIPEventOut

router = APIRouter(prefix="/calendar", tags=["calendar"])


@router.get("", response_model=CalendarResponse)
async def get_calendar(
    week: str = Query("thisweek", pattern="^(thisweek|nextweek|lastweek)$"),
    currency: str | None = Query(None, description="ISO currency code, e.g. USD"),
    impact: ImpactLevel | None = Query(None),
):
    try:
        cache = await connector_cache.get_calendar(week)
    except CacheNotReadyError:
        raise HTTPException(503, detail="Calendar cache not yet populated — try again shortly")

    events = cache.events
    if currency:
        events = [e for e in events if e.currency.upper() == currency.upper()]
    if impact:
        events = [e for e in events if e.impact == impact]

    return CalendarResponse(
        week=week,
        is_stale=cache.is_stale,
        fetched_at=cache.fetched_at,
        count=len(events),
        events=[MFIPEventOut(**e.model_dump()) for e in events],
    )


@router.get("/high-impact", response_model=CalendarResponse)
async def get_high_impact(
    week: str = Query("thisweek", pattern="^(thisweek|nextweek|lastweek)$"),
):
    try:
        cache = await connector_cache.get_calendar(week)
    except CacheNotReadyError:
        raise HTTPException(503, detail="Calendar cache not yet populated")

    events = [e for e in cache.events if e.is_high_impact]

    return CalendarResponse(
        week=week, is_stale=cache.is_stale, fetched_at=cache.fetched_at,
        count=len(events), events=[MFIPEventOut(**e.model_dump()) for e in events],
    )
