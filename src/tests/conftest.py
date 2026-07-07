"""Shared test fixtures for the Application Backend test suite."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.database.base import Base
import src.database.models  # noqa: F401 — register all ORM models


# ── In-memory SQLite engine for tests ─────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

_test_engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})

test_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ── Fake app.state ─────────────────────────────────────────────────────────────

class _FakeRollingBuffer:
    def status(self):
        return {"ready": True, "timeframes": {"M15": 800}}

    @property
    def _ready(self):
        return True


class _FakePipelineManager:
    _model_name    = "best_model"
    _feature_count = 247
    _loaded        = True
    pipeline       = object()


class _FakeScheduler:
    _running = True


class _FakeFFConnector:
    def health(self):
        return {"status": "ok", "running": True}


class _FakeEIE:
    def health(self):
        return {"status": "ok", "running": True}

    def get_active_reports(self):
        return []


class _FakeMIA:
    def health(self):
        return {"status": "ok", "running": True}

    _latest_output = None


class _FakeDFE:
    def health(self):
        return {"status": "operational", "running": True}


class _FakeWSManager:
    connection_count = 0

    async def broadcast(self, event_type, data):
        pass

    async def send_to(self, ws, event_type, data):
        pass


class FakeAppState:
    rolling_buffer   = _FakeRollingBuffer()
    pipeline_manager = _FakePipelineManager()
    scheduler        = _FakeScheduler()
    ff_connector     = _FakeFFConnector()
    eie              = _FakeEIE()
    mia              = _FakeMIA()
    dfe              = _FakeDFE()
    ws_manager       = _FakeWSManager()
    inference_engine = None


# ── ASGI test app factory ──────────────────────────────────────────────────────

def make_test_app(extra_state: dict | None = None) -> FastAPI:
    """
    Build a minimal FastAPI app wired with Application Backend services
    backed by the in-memory SQLite database.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        fake = FakeAppState()
        if extra_state:
            for k, v in extra_state.items():
                setattr(fake, k, v)
        app.state.__dict__.update(fake.__dict__)

        from src.services.settings_service import SettingsService
        app.state.settings_service = SettingsService()

        from src.services.model_registry_service import ModelRegistryService
        app.state.model_registry_service = ModelRegistryService(
            session_factory=test_session_factory, app_state=app.state
        )

        from src.services.decision_service import DecisionService
        app.state.decision_service = DecisionService(
            session_factory=test_session_factory, app_state=app.state
        )

        from src.services.history_service import HistoryService
        app.state.history_service = HistoryService(session_factory=test_session_factory)

        from src.services.notification_service import NotificationService
        app.state.notification_service = NotificationService(
            ws_manager=fake.ws_manager, session_factory=test_session_factory
        )

        from src.services.system_health_service import SystemHealthService
        app.state.system_health_service = SystemHealthService(
            app_state=app.state, start_time=time.monotonic()
        )

        from src.services.dashboard_service import DashboardService
        app.state.dashboard_service = DashboardService(
            app_state=app.state,
            session_factory=test_session_factory,
            start_time=time.monotonic(),
        )
        yield

    from src.api.v1.endpoints.dashboard       import router as dashboard_router
    from src.api.v1.endpoints.system          import router as system_router
    from src.api.v1.endpoints.settings_ep     import router as settings_router
    from src.api.v1.endpoints.models_registry import router as models_router
    from src.api.v1.endpoints.history         import router as history_router
    from src.api.v1.endpoints.auth            import router as auth_router

    app = FastAPI(lifespan=_lifespan)
    app.include_router(dashboard_router, prefix="/api/v1")
    app.include_router(system_router,    prefix="/api/v1")
    app.include_router(settings_router,  prefix="/api/v1")
    app.include_router(models_router,    prefix="/api/v1")
    app.include_router(history_router,   prefix="/api/v1")
    app.include_router(auth_router,      prefix="/api/v1")
    return app


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    app = make_test_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
