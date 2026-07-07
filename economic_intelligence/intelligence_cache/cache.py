"""
IntelligenceCache — in-memory store for EconomicIntelligenceReports.

Thread-safe via asyncio.Lock (single-process FastAPI).
Reports are keyed by event_id; the cache also maintains derived views:
  - active_events   : released events with remaining_influence > threshold
  - upcoming_events : scheduled events with timestamp_utc in the future
  - last_context    : most recent ExecutionContext snapshot
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from economic_intelligence.intelligence_models.models import EconomicIntelligenceReport
from economic_intelligence.execution_risk.calculator import ExecutionContext
from economic_intelligence.utils.config import eie_config
from economic_intelligence.utils.logger import logger


class IntelligenceCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Primary store: event_id → latest report
        self._reports:       dict[str, EconomicIntelligenceReport] = {}
        self._last_context:  Optional[ExecutionContext] = None
        self._last_cycle_at: Optional[datetime] = None
        self._cycle_count:   int = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    async def set_reports(
        self,
        reports: list[EconomicIntelligenceReport],
        context: Optional[ExecutionContext],
    ) -> None:
        async with self._lock:
            for r in reports:
                self._reports[r.event_id] = r
            if context is not None:
                self._last_context = context
            self._last_cycle_at = datetime.now(timezone.utc)
            self._cycle_count += 1
        logger.debug("EIE cache updated: {} reports total, {} new", len(self._reports), len(reports))

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_all(self) -> list[EconomicIntelligenceReport]:
        async with self._lock:
            return list(self._reports.values())

    async def get_active(self) -> list[EconomicIntelligenceReport]:
        """Released events with remaining_influence > EIE_ACTIVE_THRESHOLD."""
        async with self._lock:
            return [
                r for r in self._reports.values()
                if r.is_released and r.remaining_influence > eie_config.EIE_ACTIVE_THRESHOLD
            ]

    async def get_upcoming(self, limit_hours: float = 24.0) -> list[EconomicIntelligenceReport]:
        """Scheduled events within the next `limit_hours` hours."""
        now = datetime.now(timezone.utc)
        async with self._lock:
            result = [
                r for r in self._reports.values()
                if (
                    not r.is_released
                    and r.timestamp_utc is not None
                    and 0 <= (r.timestamp_utc - now).total_seconds() / 3600 <= limit_hours
                )
            ]
        result.sort(key=lambda r: r.timestamp_utc or datetime.max.replace(tzinfo=timezone.utc))
        return result

    async def get_high_impact_upcoming(self, window_minutes: float) -> list[EconomicIntelligenceReport]:
        """Scheduled HIGH-impact events within `window_minutes` minutes."""
        from market_intel.models.enums import ImpactLevel
        now = datetime.now(timezone.utc)
        async with self._lock:
            result = [
                r for r in self._reports.values()
                if (
                    not r.is_released
                    and r.importance == ImpactLevel.HIGH
                    and r.timestamp_utc is not None
                    and 0 <= (r.timestamp_utc - now).total_seconds() / 60 <= window_minutes
                )
            ]
        result.sort(key=lambda r: r.timestamp_utc or datetime.max.replace(tzinfo=timezone.utc))
        return result

    async def get_context(self) -> Optional[ExecutionContext]:
        async with self._lock:
            return self._last_context

    async def get_by_currency(self, currency: str) -> list[EconomicIntelligenceReport]:
        async with self._lock:
            return [r for r in self._reports.values() if r.currency == currency.upper()]

    @property
    def is_populated(self) -> bool:
        return len(self._reports) > 0

    @property
    def last_cycle_at(self) -> Optional[datetime]:
        return self._last_cycle_at

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    async def report_count(self) -> int:
        async with self._lock:
            return len(self._reports)


# Module-level singleton — shared by engine and API layer
intelligence_cache = IntelligenceCache()
