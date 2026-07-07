"""Repository for runtime application settings."""
from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.app_settings import AppSettings
from src.database.repositories.base import BaseRepository


class SettingsRepository(BaseRepository[AppSettings]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(AppSettings, session)

    async def get_by_key(self, key: str) -> AppSettings | None:
        result = await self.session.execute(
            select(AppSettings).where(AppSettings.key == key)
        )
        return result.scalar_one_or_none()

    async def get_by_category(self, category: str) -> Sequence[AppSettings]:
        result = await self.session.execute(
            select(AppSettings)
            .where(AppSettings.category == category)
            .order_by(AppSettings.key)
        )
        return result.scalars().all()

    async def list_all(self) -> Sequence[AppSettings]:
        result = await self.session.execute(
            select(AppSettings).order_by(AppSettings.category, AppSettings.key)
        )
        return result.scalars().all()

    async def upsert(
        self,
        *,
        key: str,
        value: str,
        value_type: str = "string",
        category: str = "general",
        description: str | None = None,
        is_secret: bool = False,
    ) -> AppSettings:
        existing = await self.get_by_key(key)
        if existing:
            existing.value = value
            existing.value_type = value_type
            await self.session.flush()
            return existing

        setting = AppSettings(
            key=key,
            value=value,
            value_type=value_type,
            category=category,
            description=description,
            is_secret=is_secret,
        )
        return await self.add(setting)
