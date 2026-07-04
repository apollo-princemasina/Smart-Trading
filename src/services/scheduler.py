"""MFIP Scheduler — fires inference on every M15 bar close."""
from __future__ import annotations

import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron      import CronTrigger
from loguru import logger

from src.api.core.config import settings


class MFIPScheduler:
    def __init__(self, prediction_service) -> None:
        self._svc       = prediction_service
        self._scheduler = AsyncIOScheduler(timezone="UTC")

    def start(self) -> None:
        trigger = CronTrigger.from_crontab(settings.SCHEDULER_M15_CRON, timezone="UTC")
        self._scheduler.add_job(
            self._tick,
            trigger=trigger,
            id="m15_inference",
            name="M15 bar close inference",
            max_instances=1,
            misfire_grace_time=60,
        )
        self._scheduler.start()
        logger.info("Scheduler started — next run at {}", self._next_run())

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._scheduler.running

    def _next_run(self) -> str:
        jobs = self._scheduler.get_jobs()
        if jobs:
            return str(jobs[0].next_run_time)
        return "unknown"

    async def _tick(self) -> None:
        logger.debug("Scheduler tick — running M15 inference cycle")
        try:
            await self._svc.run_cycle()
        except Exception:
            logger.exception("Scheduler tick failed")
