import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from market_intel.models.event import MFIPEvent
from ..utils.logger import logger


@dataclass
class WeekCache:
    events:     list[MFIPEvent]
    fetched_at: datetime
    etag:       Optional[str] = None
    is_stale:   bool = False


class ConnectorCache:
    """
    In-memory store for all connector data.

    The API layer reads exclusively from this cache — it never triggers I/O.
    Scheduler jobs are the only writers.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._weeks: dict[str, WeekCache] = {}

    async def set_calendar(self, week: str, events: list[MFIPEvent], etag: Optional[str] = None) -> None:
        async with self._lock:
            self._weeks[week] = WeekCache(
                events=events,
                fetched_at=datetime.now(timezone.utc),
                etag=etag,
            )
            logger.debug(f"Cache updated: {week} ({len(events)} MFIPEvents)")

    async def get_calendar(self, week: str) -> WeekCache:
        async with self._lock:
            entry = self._weeks.get(week)
        if entry is None:
            raise CacheNotReadyError(f"Calendar cache for '{week}' not yet populated")
        return entry

    async def mark_stale(self, week: str) -> None:
        async with self._lock:
            if week in self._weeks:
                self._weeks[week].is_stale = True

    def is_populated(self, week: str) -> bool:
        return week in self._weeks


class CacheNotReadyError(RuntimeError):
    pass


connector_cache = ConnectorCache()
