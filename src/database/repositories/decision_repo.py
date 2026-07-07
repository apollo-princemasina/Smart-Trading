"""Repository for decision history queries."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.decision_history import DecisionHistory
from src.database.repositories.base import BaseRepository


class DecisionRepository(BaseRepository[DecisionHistory]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(DecisionHistory, session)

    async def get_by_decision_id(self, decision_id: str) -> DecisionHistory | None:
        result = await self.session.execute(
            select(DecisionHistory).where(DecisionHistory.decision_id == decision_id)
        )
        return result.scalar_one_or_none()

    async def latest(self) -> DecisionHistory | None:
        result = await self.session.execute(
            select(DecisionHistory)
            .order_by(DecisionHistory.generated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_paginated(
        self,
        *,
        recommendation: str | None = None,
        strength: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[Sequence[DecisionHistory], int]:
        stmt = select(DecisionHistory)

        if recommendation:
            stmt = stmt.where(DecisionHistory.recommendation == recommendation.upper())
        if strength:
            stmt = stmt.where(DecisionHistory.strength == strength.upper())
        if after:
            stmt = stmt.where(DecisionHistory.generated_at >= after)
        if before:
            stmt = stmt.where(DecisionHistory.generated_at <= before)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total: int = (await self.session.execute(count_stmt)).scalar_one()

        stmt = (
            stmt.order_by(DecisionHistory.generated_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return rows, total

    async def exists(self, decision_id: str) -> bool:
        result = await self.session.execute(
            select(func.count())
            .select_from(DecisionHistory)
            .where(DecisionHistory.decision_id == decision_id)
        )
        return result.scalar_one() > 0
