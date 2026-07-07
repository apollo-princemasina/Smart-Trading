"""SettingsService — typed CRUD over the app_settings table."""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.database.repositories.settings_repo import SettingsRepository


_TYPE_CASTERS = {
    "bool":   lambda v: v.lower() in ("true", "1", "yes"),
    "int":    int,
    "float":  float,
    "json":   json.loads,
    "string": str,
}


class SettingsService:
    """
    Reads and writes application settings from the DB.

    Values are stored as text; value_type determines how they are cast
    when read back. Secrets are redacted in public API responses.
    """

    def __init__(self) -> None:
        # In-memory cache populated on first access per request
        self._cache: dict[str, Any] = {}
        self._cache_populated = False

    # ── Read ──────────────────────────────────────────────────────────────

    async def get(
        self, key: str, session_factory
    ) -> Any | None:
        async with session_factory() as session:
            repo = SettingsRepository(session)
            row = await repo.get_by_key(key)
            if row is None:
                return None
            return self._cast(row.value, row.value_type)

    async def get_all(self, session_factory) -> dict[str, Any]:
        async with session_factory() as session:
            repo = SettingsRepository(session)
            rows = await repo.list_all()
            return {
                r.key: self._cast(r.value, r.value_type) if not r.is_secret else "***"
                for r in rows
            }

    async def get_by_category(
        self, category: str, session_factory
    ) -> dict[str, Any]:
        async with session_factory() as session:
            repo = SettingsRepository(session)
            rows = await repo.get_by_category(category)
            return {
                r.key: self._cast(r.value, r.value_type) if not r.is_secret else "***"
                for r in rows
            }

    async def get_all_rows(self, session_factory):
        """Return raw AppSettings rows (for API serialisation with metadata)."""
        async with session_factory() as session:
            repo = SettingsRepository(session)
            return await repo.list_all()

    # ── Write ─────────────────────────────────────────────────────────────

    async def set(
        self,
        key: str,
        value: Any,
        *,
        session_factory,
        value_type: str = "string",
        category: str = "general",
        description: str | None = None,
        is_secret: bool = False,
    ) -> None:
        str_value = self._to_str(value, value_type)
        async with session_factory() as session:
            repo = SettingsRepository(session)
            await repo.upsert(
                key=key,
                value=str_value,
                value_type=value_type,
                category=category,
                description=description,
                is_secret=is_secret,
            )
            await session.commit()
        logger.info("Setting updated: {}={!r}", key, value if not is_secret else "***")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _cast(value: str, value_type: str) -> Any:
        caster = _TYPE_CASTERS.get(value_type, str)
        try:
            return caster(value)
        except (ValueError, TypeError):
            return value

    @staticmethod
    def _to_str(value: Any, value_type: str) -> str:
        if value_type == "json":
            return json.dumps(value)
        return str(value)
