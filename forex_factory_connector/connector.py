"""
ForexFactoryConnector — top-level lifecycle manager.

Owns startup (cache warm-up + scheduler), shutdown (scheduler stop + HTTP teardown),
and exposes the connector's operational state to the health endpoint.

Usage inside FastAPI lifespan
------------------------------
    connector = ForexFactoryConnector()
    await connector.startup()
    app.state.ff_connector = connector
    yield
    await connector.shutdown()
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .fetcher.http_client          import HTTPClient
from .scheduler                    import build_scheduler
from .scheduler.jobs.calendar_job  import run_calendar_job
from .scheduler.health_reporter    import health, SCHEMA_VERSION
from .utils.logger                 import logger
from .utils.config                 import settings


class ForexFactoryConnector:
    SCHEMA_VERSION = SCHEMA_VERSION
    PROVIDER       = "forex_factory"

    def __init__(self) -> None:
        self._scheduler:  Optional[AsyncIOScheduler] = None
        self._started_at: Optional[datetime]         = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """
        Called once from the FastAPI lifespan on application startup.

        Step 1: populate the cache immediately (all three weeks in parallel).
        Step 2: start the APScheduler for continuous background polling.

        The two-step design means the very first API request is always served
        from cache — the scheduler never has to "catch up" on the first request.
        """
        logger.info("ForexFactoryConnector: starting up")

        # Build scheduler first — this registers all jobs with the health reporter
        # so that warm-up calls to record_success/record_failure are persisted.
        self._scheduler = build_scheduler()

        # Warm cache — failures are logged but do NOT abort startup
        logger.info("ForexFactoryConnector: warming cache (thisweek / nextweek / lastweek)…")
        results = await asyncio.gather(
            run_calendar_job("thisweek"),
            run_calendar_job("nextweek"),
            run_calendar_job("lastweek"),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.warning(
                f"ForexFactoryConnector: {len(errors)}/3 warm-up fetches failed — "
                f"endpoints will serve 503 until the scheduler retries"
            )
        else:
            logger.info("ForexFactoryConnector: cache warm-up complete")

        # Start background polling
        self._scheduler.start()
        self._started_at = datetime.now(timezone.utc)
        logger.info("ForexFactoryConnector: scheduler started — polling active")

    async def shutdown(self) -> None:
        """
        Called once from the FastAPI lifespan on application shutdown.

        Stops the APScheduler (without waiting for running jobs to drain —
        they are idempotent) and closes the shared HTTP client session.
        """
        logger.info("ForexFactoryConnector: shutting down")

        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("ForexFactoryConnector: scheduler stopped")

        await HTTPClient.close()
        logger.info("ForexFactoryConnector: HTTP client closed")
        logger.info("ForexFactoryConnector: shutdown complete")

    # ── State accessors ───────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return bool(self._scheduler and self._scheduler.running)

    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at

    @property
    def uptime_s(self) -> Optional[float]:
        if not self._started_at:
            return None
        return round((datetime.now(timezone.utc) - self._started_at).total_seconds(), 1)

    def get_scheduler(self) -> Optional[AsyncIOScheduler]:
        return self._scheduler

    def next_run(self, job_id: str) -> Optional[datetime]:
        """Return the APScheduler next_run_time for a given job ID (UTC-aware)."""
        if not self._scheduler:
            return None
        job = self._scheduler.get_job(job_id)
        return job.next_run_time if job else None
