"""Repository for system event log queries."""
from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.system_log import SystemLog
from src.database.repositories.base import BaseRepository


class SystemLogRepository(BaseRepository[SystemLog]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(SystemLog, session)

    async def recent(
        self,
        *,
        limit: int = 100,
        level: str | None = None,
        component: str | None = None,
        event_type: str | None = None,
    ) -> Sequence[SystemLog]:
        stmt = select(SystemLog)
        if level:
            stmt = stmt.where(SystemLog.level == level.upper())
        if component:
            stmt = stmt.where(SystemLog.component == component)
        if event_type:
            stmt = stmt.where(SystemLog.event_type == event_type)
        stmt = stmt.order_by(SystemLog.logged_at.desc()).limit(limit)
        return (await self.session.execute(stmt)).scalars().all()

    async def log(
        self,
        *,
        level: str,
        component: str,
        event_type: str,
        message: str,
        details: dict | None = None,
        correlation_id: str | None = None,
    ) -> SystemLog:
        entry = SystemLog(
            level=level.upper(),
            component=component,
            event_type=event_type,
            message=message,
            details=details,
            correlation_id=correlation_id,
        )
        return await self.add(entry)
