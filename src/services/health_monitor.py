"""Health Monitor — aggregates system health into a single report."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import text

from src.api.core.config    import settings
from src.api.schemas.health import (
    BufferStatus, DBStatus, HealthResponse, ModelStatus,
)


class HealthMonitor:
    def __init__(
        self,
        rolling_buffer,
        pipeline_manager,
        session_factory,
        start_time: float,
    ) -> None:
        self._buffer   = rolling_buffer
        self._pipeline = pipeline_manager
        self._factory  = session_factory
        self._start    = start_time

    async def check(self) -> HealthResponse:
        buffer_statuses = self._check_buffers()
        model_status    = self._check_model()
        db_status       = await self._check_db()

        all_ready = (
            all(b.ready for b in buffer_statuses)
            and model_status.loaded
            and db_status.connected
        )
        any_buffer_ready = any(b.ready for b in buffer_statuses)

        if all_ready:
            status = "healthy"
        elif any_buffer_ready or model_status.loaded:
            status = "degraded"
        else:
            status = "unhealthy"

        from src.api.core.config import settings as s
        return HealthResponse(
            status=status,
            version=s.APP_VERSION,
            environment=s.APP_ENV,
            timestamp=datetime.now(timezone.utc),
            uptime_s=time.monotonic() - self._start,
            buffer=buffer_statuses,
            model=model_status,
            database=db_status,
            scheduler_running=True,  # checked by scheduler itself
        )

    def _check_buffers(self) -> list[BufferStatus]:
        statuses = []
        for tf, expected in settings.buffer_sizes.items():
            candles = self._buffer.get_candles(tf)
            ready   = self._buffer.is_ready(tf)
            oldest  = candles[0].get("timestamp")  if candles else None
            newest  = candles[-1].get("timestamp") if candles else None
            statuses.append(BufferStatus(
                timeframe=tf,
                size=len(candles),
                expected=expected,
                oldest_bar=oldest,
                newest_bar=newest,
                ready=ready,
            ))
        return statuses

    def _check_model(self) -> ModelStatus:
        return ModelStatus(
            loaded=self._pipeline.is_loaded,
            bundle_path=str(self._pipeline.bundle_dir),
            feature_count=self._pipeline.feature_count,
            model_name=self._pipeline.model_name,
            loaded_at=self._pipeline.loaded_at,
        )

    async def _check_db(self) -> DBStatus:
        try:
            async with self._factory() as session:
                await session.execute(text("SELECT 1"))
                from sqlalchemy import func, select
                from src.database.models.prediction import Prediction
                result = await session.execute(select(func.count()).select_from(Prediction))
                count = result.scalar_one()
            return DBStatus(connected=True, prediction_count=count)
        except Exception as e:
            logger.warning("DB health check failed: {}", e)
            return DBStatus(connected=False, prediction_count=0)
