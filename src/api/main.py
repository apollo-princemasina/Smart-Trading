"""MFIP FastAPI application entry point.

Start locally:
    uvicorn src.api.main:app --reload --port 8000

Or via Docker Compose:
    docker compose up backend
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env into os.environ early — before any os.getenv()-based config modules are imported
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.core.config      import settings
from src.api.core.exceptions  import register_exception_handlers
from src.api.core.logging     import setup_logging
from src.api.v1.router        import v1_router
from src.api.websocket.manager import WebSocketManager
from src.middleware.request_id import RequestIDMiddleware


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
        app_state=app.state,
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

    # Market Intelligence Layer — Forex Factory Connector
    from forex_factory_connector.connector import ForexFactoryConnector
    app.state.ff_connector = ForexFactoryConnector()
    await app.state.ff_connector.startup()
    logger.info("Market Intelligence Layer ready")

    # Economic Intelligence Engine — Phase 3
    from economic_intelligence.engine import EconomicIntelligenceEngine
    app.state.eie = EconomicIntelligenceEngine()
    await app.state.eie.startup()
    logger.info("Economic Intelligence Engine ready")

    # Market Intelligence AI Layer — Phase 4
    from market_intelligence_ai.engine import MarketIntelligenceAIEngine
    app.state.mia = MarketIntelligenceAIEngine()
    await app.state.mia.startup()
    logger.info("Market Intelligence AI Layer ready")

    # Decision Fusion Engine — Phase 5
    from decision_fusion.engine import DecisionFusionEngine
    app.state.dfe = DecisionFusionEngine()
    await app.state.dfe.startup()
    logger.info("Decision Fusion Engine ready")

    # ── Application Backend services — Phase 6 ────────────────────────────
    from src.database.session import async_session_factory

    from src.services.settings_service import SettingsService
    app.state.settings_service = SettingsService()
    logger.info("Settings service ready")

    from src.services.model_registry_service import ModelRegistryService
    app.state.model_registry_service = ModelRegistryService(
        session_factory=async_session_factory,
        app_state=app.state,
    )
    await app.state.model_registry_service.register_from_pipeline_manager()
    logger.info("Model Registry service ready")

    from src.services.decision_service import DecisionService
    app.state.decision_service = DecisionService(
        session_factory=async_session_factory,
        app_state=app.state,
    )
    logger.info("Decision service ready")

    from src.services.history_service import HistoryService
    app.state.history_service = HistoryService(session_factory=async_session_factory)
    logger.info("History service ready")

    from src.services.notification_service import NotificationService
    app.state.notification_service = NotificationService(
        ws_manager=app.state.ws_manager,
        session_factory=async_session_factory,
    )
    logger.info("Notification service ready")

    from src.services.system_health_service import SystemHealthService
    app.state.system_health_service = SystemHealthService(
        app_state=app.state,
        start_time=time.monotonic(),
    )
    logger.info("System health service ready")

    from src.services.dashboard_service import DashboardService
    app.state.dashboard_service = DashboardService(
        app_state=app.state,
        session_factory=async_session_factory,
        start_time=time.monotonic(),
    )
    logger.info("Dashboard service ready")

    # Wire MIA to live market context so Groq sees current price + regime
    def _mia_market_context():
        import datetime as _dt
        ctx = {}
        try:
            rb = getattr(app.state, "rolling_buffer", None)
            if rb:
                ctx["current_price"] = rb.latest_close("M15")
        except Exception:
            pass
        try:
            ie = getattr(app.state, "inference_engine", None)
            if ie:
                regime = ie.latest_regime()
                if regime:
                    ctx["regime"]           = regime.dominant_regime
                    ctx["atr_pips"]         = regime.atr_pips
                    ctx["adx"]              = regime.adx
                    ctx["regime_narrative"] = regime.narrative
                result = ie.latest_result()
                if result:
                    ctx["session"]            = result.get("session")
                    ctx["latest_direction"]   = result.get("direction")
                    ctx["latest_confidence"]  = result.get("confidence")
                    ctx["prob_buy"]           = result.get("prob_buy")
                    ctx["prob_sell"]          = result.get("prob_sell")
                    ctx["prob_hold"]          = result.get("prob_hold")
        except Exception:
            pass
        # Inject upcoming EUR/USD news from Forex Factory for EURUSD-specific analysis
        try:
            from forex_factory_connector.cache.memory_cache import connector_cache
            week_cache = connector_cache._weeks.get("thisweek")
            if week_cache and week_cache.events:
                now = _dt.datetime.now(_dt.timezone.utc)
                cutoff = now + _dt.timedelta(hours=12)
                relevant = [
                    e for e in week_cache.events
                    if getattr(e, "timestamp_utc", None)
                    and now <= e.timestamp_utc <= cutoff
                    and getattr(e, "currency", "") in ("EUR", "USD")
                    and getattr(e, "impact", "").upper() in ("HIGH", "MEDIUM")
                ]
                relevant.sort(key=lambda e: e.timestamp_utc)
                ctx["ff_upcoming_news"] = [
                    {
                        "title":    ev.title,
                        "currency": ev.currency,
                        "impact":   ev.impact if isinstance(ev.impact, str) else ev.impact.value,
                        "time":     ev.timestamp_utc.strftime("%H:%M UTC"),
                        "forecast": getattr(ev, "forecast", None),
                        "previous": getattr(ev, "previous", None),
                        "actual":   getattr(ev, "actual", None),
                    }
                    for ev in relevant[:8]
                ]
        except Exception:
            pass
        return ctx

    app.state.mia.set_market_context_provider(_mia_market_context)
    logger.info("MIA market context provider wired")

    logger.info("=== MFIP ready — http://{}:{} ===", settings.APP_HOST, settings.APP_PORT)

    # On startup: update buffer then catch up any M15 bars missed while offline.
    # This prevents being "N bars behind" after a restart mid-session.
    import asyncio as _asyncio
    import pandas as _pd
    from datetime import timezone as _tz

    async def _startup_catchup():
        await _asyncio.sleep(1)

        rb  = app.state.rolling_buffer
        svc = app.state.prediction_service
        eng = app.state.inference_engine

        # 1. Pull the latest closed bar from Deriv before doing anything.
        #    Without this, the buffer reflects the last shutdown state and may
        #    be missing bars that closed while the backend was offline.
        try:
            await rb.update()
            logger.info("Startup: buffer refreshed from Deriv")
        except Exception:
            logger.exception("Startup buffer refresh failed — proceeding with cached data")

        # 2. Find the last prediction we stored for this symbol.
        from src.database.repositories.prediction_repo import PredictionRepository
        from src.database.session import async_session_factory as _sf
        async with _sf() as _db:
            _repo = PredictionRepository(_db)
            _latest_pred = await _repo.latest(settings.MODEL_SYMBOL)
        last_ts = _latest_pred.signal_time if _latest_pred else None

        # 3. Identify all M15 bars in the buffer that are newer than last_ts.
        m15_df = rb.as_dataframe("M15")
        if m15_df.empty:
            logger.warning("Startup catchup: M15 buffer empty — skipping")
            return

        if last_ts is not None:
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=_tz.utc)
            missed_df = m15_df[m15_df["timestamp"] > _pd.Timestamp(last_ts)]
        else:
            missed_df = m15_df.iloc[-1:]  # first ever run — process latest bar only

        htf_dfs = {tf: rb.as_dataframe(tf) for tf in ("H1", "H4", "D1", "W1")}

        # 4. Process each missed bar oldest-first.
        processed = 0
        if not missed_df.empty:
            logger.info(
                "Startup catchup: processing {} missed bar(s) since {}",
                len(missed_df), last_ts,
            )
            for bar_ts in missed_df["timestamp"]:
                slice_df = m15_df[m15_df["timestamp"] <= bar_ts].copy()
                result = await eng.run_inference_at(slice_df, htf_dfs)
                if result:
                    await svc._persist(result)
                    processed += 1
            logger.info("Startup catchup complete — {} bar(s) processed", processed)
        else:
            logger.info("Startup catchup: no missed bars since {} — triggering fresh inference", last_ts)

        # 5. Always broadcast latest result and trigger DFE on every startup,
        #    even if no bars were missed — ensures dashboard shows current state.
        latest = eng.latest_result()
        if latest is None:
            # Engine hasn't run yet this session — run inference on the current bar
            result = await eng.run_inference_at(m15_df, htf_dfs)
            if result:
                await svc._persist(result)
                latest = result

        if latest:
            try:
                await svc._broadcast(latest)
                await svc._trigger_dfe(latest)
                logger.info("Startup catchup: dashboard updated with latest signal")
            except Exception:
                logger.exception("Startup catchup broadcast failed")

    _asyncio.create_task(_startup_catchup())
    logger.info("Startup catch-up task queued (fires in 1 s)")

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("MFIP shutting down...")

    # Decision Fusion Engine — shut down first (depends on MIA/EIE)
    if hasattr(app.state, "dfe"):
        await app.state.dfe.shutdown()

    # Market Intelligence AI Layer — shut down before EIE
    if hasattr(app.state, "mia"):
        await app.state.mia.shutdown()

    # Economic Intelligence Engine — shut down before connector
    if hasattr(app.state, "eie"):
        await app.state.eie.shutdown()

    # Market Intelligence Layer — graceful connector teardown before buffer save
    if hasattr(app.state, "ff_connector"):
        await app.state.ff_connector.shutdown()

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

    # Middleware — order matters: outermost runs first on request, last on response.
    # RequestID wraps everything so correlation IDs are available to all layers.
    app.add_middleware(RequestIDMiddleware)

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

        # Send connection acknowledgement with current snapshot
        try:
            from src.api.websocket.events import WSEventType
            await manager.send_to(ws, WSEventType.CONNECTION_ACK, {"status": "connected"})
        except Exception:
            pass

        try:
            while True:
                # Route client messages (subscribe, ping, etc.)
                raw = await ws.receive_text()
                await manager.handle_message(ws, raw)
        except WebSocketDisconnect:
            await manager.disconnect(ws)

    # Exception handlers
    register_exception_handlers(app)

    return app


app = create_app()
