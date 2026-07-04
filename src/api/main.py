"""MFIP FastAPI application entry point.

Start locally:
    uvicorn src.api.main:app --reload --port 8000

Or via Docker Compose:
    docker compose up backend
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.core.config     import settings
from src.api.core.exceptions  import register_exception_handlers
from src.api.core.logging     import setup_logging
from src.api.v1.router        import v1_router
from src.api.websocket.manager import WebSocketManager


# ── Startup / shutdown lifecycle ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ── STARTUP ──────────────────────────────────────────────────────────────
    setup_logging(level=settings.LOG_LEVEL, json=settings.is_production)
    logger.info("=== MFIP {} starting (env={}) ===", settings.APP_VERSION, settings.APP_ENV)

    # Database
    from src.database.session import create_all_tables
    await create_all_tables()
    logger.info("Database tables ensured")

    # WebSocket manager
    app.state.ws_manager = WebSocketManager()

    # Pipeline manager (loads model bundle once — singleton)
    from src.services.pipeline_manager import PipelineManager
    app.state.pipeline_manager = PipelineManager()
    app.state.pipeline_manager.load()
    logger.info("Model bundle loaded from {}", settings.MODEL_BUNDLE_DIR)

    # Rolling buffer manager
    from src.services.rolling_buffer import RollingBufferManager
    app.state.rolling_buffer = RollingBufferManager()
    await app.state.rolling_buffer.initialise()
    logger.info("Rolling buffers populated")

    # Inference engine (wraps ML inference modules)
    from src.services.inference_engine import InferenceEngine
    app.state.inference_engine = InferenceEngine(
        pipeline_manager=app.state.pipeline_manager,
        rolling_buffer=app.state.rolling_buffer,
    )

    # Prediction service (orchestrates inference → DB → WebSocket)
    from src.services.prediction_service import PredictionService
    from src.database.session import async_session_factory
    app.state.prediction_service = PredictionService(
        inference_engine=app.state.inference_engine,
        session_factory=async_session_factory,
        ws_manager=app.state.ws_manager,
    )

    # Health monitor
    from src.services.health_monitor import HealthMonitor
    app.state.health_monitor = HealthMonitor(
        rolling_buffer=app.state.rolling_buffer,
        pipeline_manager=app.state.pipeline_manager,
        session_factory=async_session_factory,
        start_time=time.monotonic(),
    )

    # Scheduler — fires PredictionService on every M15 bar close
    from src.services.scheduler import MFIPScheduler
    app.state.scheduler = MFIPScheduler(
        prediction_service=app.state.prediction_service,
    )
    app.state.scheduler.start()
    logger.info("Scheduler started (cron={})", settings.SCHEDULER_M15_CRON)

    # Run one inference cycle immediately (no buffer update — buffers just initialized)
    # This ensures the regime panel and signal panel are populated on first page load
    # without burning API rate-limit quota.
    import asyncio

    async def _initial_inference():
        import asyncio as _asyncio
        await _asyncio.sleep(2)  # Let the event loop finish startup before inferring
        await app.state.prediction_service.run_cycle(update_buffers=False)

    asyncio.create_task(_initial_inference())
    logger.info("Initial inference cycle queued (fires in 2 s)")

    logger.info("=== MFIP ready — http://{}:{} ===", settings.APP_HOST, settings.APP_PORT)

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("MFIP shutting down...")
    app.state.scheduler.stop()

    # Persist buffer cache so cold restarts are fast
    if settings.BUFFER_CACHE_ENABLED:
        await app.state.rolling_buffer.save_cache()
        logger.info("Buffer cache saved")

    logger.info("MFIP shutdown complete")


# ── App construction ──────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="MFIP — Moonshot Forex Intelligence Platform",
        description=(
            "AI-powered live inference API for EURUSD M15. "
            "Provides BUY/SELL/HOLD signals with ICT/SMC market regime analysis."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # CORS — allow the Next.js dev server and production URL
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(v1_router)

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        manager: WebSocketManager = ws.app.state.ws_manager
        await manager.connect(ws)
        try:
            while True:
                # Keep connection alive — client can send pings
                await ws.receive_text()
        except WebSocketDisconnect:
            await manager.disconnect(ws)

    # Exception handlers
    register_exception_handlers(app)

    return app


app = create_app()
