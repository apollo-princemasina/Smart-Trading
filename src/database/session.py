"""Async SQLAlchemy session factory.

Supports both PostgreSQL (asyncpg) and SQLite (aiosqlite) so local
development without Docker works out of the box.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.core.config import settings

_is_postgres = "postgresql" in settings.DATABASE_URL

_engine_kwargs: dict = {"echo": settings.APP_DEBUG}
if _is_postgres:
    # PostgreSQL supports connection pooling
    _engine_kwargs.update({"pool_pre_ping": True, "pool_size": 5, "max_overflow": 10})
else:
    # SQLite (aiosqlite) uses a single connection — no pool args
    _engine_kwargs.update({"connect_args": {"check_same_thread": False}})

_engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def create_all_tables() -> None:
    """Create all tables that don't exist yet (dev convenience, not for prod migrations)."""
    from src.database.base import Base  # noqa: F401 — registers all models
    import src.database.models  # noqa: F401 — ensure models are imported

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_engine():
    return _engine
