"""FastAPI dependency injection providers.

Usage in endpoints:
    from src.api.core.dependencies import get_db, get_prediction_service

    @router.get("/predictions")
    async def list_predictions(
        db: AsyncSession = Depends(get_db),
        svc: PredictionService = Depends(get_prediction_service),
    ):
        ...
"""
from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import async_session_factory


# ── Database session ──────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a per-request async database session, auto-committed on success."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Service providers (populated after app startup) ───────────────────────────
# Services are singletons attached to app.state during lifespan startup.
# These dependency functions retrieve them, making them injectable in routes.

from fastapi import Request  # noqa: E402 — import after stdlib to avoid circular


def get_rolling_buffer(request: Request):
    return request.app.state.rolling_buffer


def get_pipeline_manager(request: Request):
    return request.app.state.pipeline_manager


def get_inference_engine(request: Request):
    return request.app.state.inference_engine


def get_prediction_service(request: Request):
    return request.app.state.prediction_service


def get_scheduler(request: Request):
    return request.app.state.scheduler


def get_health_monitor(request: Request):
    return request.app.state.health_monitor
