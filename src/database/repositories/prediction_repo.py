"""Prediction-specific repository."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.prediction import Prediction
from src.database.models.outcome    import PredictionOutcome
from src.database.repositories.base import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Prediction, session)

    async def exists_for_signal_time(self, signal_time: datetime, symbol: str = "EURUSD") -> bool:
        result = await self.session.execute(
            select(Prediction.id)
            .where(Prediction.symbol == symbol)
            .where(Prediction.signal_time == signal_time)
            .limit(1)
        )
        return result.scalars().first() is not None

    async def latest(self, symbol: str = "EURUSD") -> Prediction | None:
        result = await self.session.execute(
            select(Prediction)
            .where(Prediction.symbol == symbol)
            .order_by(desc(Prediction.signal_time))
            .limit(1)
        )
        return result.scalars().first()

    async def list_recent(
        self,
        symbol:    str = "EURUSD",
        direction: str | None = None,
        page:      int = 1,
        page_size: int = 20,
    ) -> tuple[Sequence[Prediction], int]:
        """Return (rows, total_count)."""
        base = select(Prediction).where(Prediction.symbol == symbol)
        if direction:
            base = base.where(Prediction.direction == direction.upper())

        from sqlalchemy import func
        count_result = await self.session.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar_one()

        rows_result = await self.session.execute(
            base.order_by(desc(Prediction.signal_time))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return rows_result.scalars().all(), total

    async def get_pending_outcomes(self) -> Sequence[Prediction]:
        """Return predictions that don't yet have a resolved outcome."""
        result = await self.session.execute(
            select(Prediction)
            .outerjoin(PredictionOutcome, Prediction.id == PredictionOutcome.prediction_id)
            .where(
                (PredictionOutcome.id == None) |  # noqa: E711
                (PredictionOutcome.outcome == "PENDING")
            )
            .where(Prediction.direction != "HOLD")
            .order_by(desc(Prediction.signal_time))
            .limit(100)
        )
        return result.scalars().all()
