from fastapi import APIRouter, HTTPException
from ...cache.memory_cache import connector_cache, CacheNotReadyError
from ..schemas import CalendarResponse, MFIPEventOut

router = APIRouter(prefix="/speeches", tags=["speeches"])


@router.get("", response_model=CalendarResponse)
async def get_speeches():
    try:
        cache = await connector_cache.get_calendar("thisweek")
    except CacheNotReadyError:
        raise HTTPException(503, detail="Calendar cache not yet populated")

    # is_speech is computed at normalization time — no keyword scan here
    speeches = [e for e in cache.events if e.is_speech]

    return CalendarResponse(
        week="thisweek", is_stale=cache.is_stale, fetched_at=cache.fetched_at,
        count=len(speeches), events=[MFIPEventOut(**e.model_dump()) for e in speeches],
    )
