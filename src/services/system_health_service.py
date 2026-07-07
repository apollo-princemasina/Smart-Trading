"""SystemHealthService — deep health check across all MFIP engines."""
from __future__ import annotations

import time
from typing import Any

from loguru import logger


class SystemHealthService:
    """
    Aggregates health from every Phase 1–5 engine plus infrastructure.

    All checks are non-destructive reads from app.state — no engine logic
    is replicated here.
    """

    def __init__(self, app_state: Any, start_time: float) -> None:
        self._state = app_state
        self._start_time = start_time

    async def full_health(self) -> dict:
        uptime_s = time.monotonic() - self._start_time
        components: dict[str, dict] = {}

        # ── Rolling Buffer ────────────────────────────────────────────────
        try:
            rb = self._state.rolling_buffer
            ready = getattr(rb, "all_ready", False)
            tf_status = {tf: rb.is_ready(tf) for tf in ("M15", "H1", "H4", "D1", "W1")}
            components["rolling_buffer"] = {
                "status":     "ok" if ready else "degraded",
                "ready":      ready,
                "timeframes": tf_status,
            }
        except Exception as exc:
            components["rolling_buffer"] = {"status": "error", "error": str(exc)}

        # ── ML Pipeline ───────────────────────────────────────────────────
        try:
            pm = self._state.pipeline_manager
            loaded = getattr(pm, "_loaded", False) or getattr(pm, "pipeline", None) is not None
            components["ml_pipeline"] = {
                "status": "ok" if loaded else "degraded",
                "model_loaded": loaded,
                "model_name": getattr(pm, "_model_name", None),
                "feature_count": getattr(pm, "_feature_count", None),
            }
        except Exception as exc:
            components["ml_pipeline"] = {"status": "error", "error": str(exc)}

        # ── Scheduler ─────────────────────────────────────────────────────
        try:
            scheduler = self._state.scheduler
            running = getattr(scheduler, "_running", True)
            components["scheduler"] = {
                "status": "ok" if running else "stopped",
                "running": running,
            }
        except Exception as exc:
            components["scheduler"] = {"status": "error", "error": str(exc)}

        # ── Forex Factory Connector ───────────────────────────────────────
        try:
            ff = self._state.ff_connector
            running = getattr(ff, "is_running", False)
            uptime  = getattr(ff, "uptime_s", None)
            # Check job health from the module-level health reporter
            try:
                from forex_factory_connector.scheduler.health_reporter import health as ff_health_reporter
                job_statuses = {jid: m.status for jid, m in ff_health_reporter.all_jobs().items()}
                has_any_success = any(
                    m.last_success is not None
                    for m in ff_health_reporter.all_jobs().values()
                )
                # Only degrade on circuit-open or recently-regressed jobs.
                # "not_started" = 404/weekend unavailability (expected); "initializing" = just started.
                has_problems = any(
                    m.status in ("down", "degraded")
                    for m in ff_health_reporter.all_jobs().values()
                )
            except Exception:
                job_statuses = {}
                has_any_success = running
                has_problems = not running
            components["forex_factory"] = {
                "status":    "ok" if (running and not has_problems) else "degraded",
                "running":   running,
                "uptime_s":  uptime,
                "jobs":      job_statuses,
                "data_available": has_any_success,
            }
        except Exception as exc:
            components["forex_factory"] = {"status": "error", "error": str(exc)}

        # ── Economic Intelligence Engine ───────────────────────────────────
        try:
            eie = self._state.eie
            running = getattr(eie, "is_running", False)
            try:
                from economic_intelligence.intelligence_cache.cache import intelligence_cache
                active_count = len(await intelligence_cache.get_active())
            except Exception:
                active_count = 0
            components["economic_intelligence"] = {
                "status":       "ok" if running else "degraded",
                "running":      running,
                "active_events": active_count,
            }
        except Exception as exc:
            components["economic_intelligence"] = {"status": "error", "error": str(exc)}

        # ── Market Intelligence AI ─────────────────────────────────────────
        try:
            mia = self._state.mia
            mia_health = mia.health() if hasattr(mia, "health") else {}
            components["market_intelligence_ai"] = {
                "status": "ok" if mia_health.get("running", False) else "degraded",
                **mia_health,
            }
        except Exception as exc:
            components["market_intelligence_ai"] = {"status": "error", "error": str(exc)}

        # ── Decision Fusion Engine ─────────────────────────────────────────
        try:
            dfe = self._state.dfe
            dfe_health = dfe.health() if hasattr(dfe, "health") else {}
            is_operational = dfe_health.get("running", False) or dfe_health.get("status") == "operational"
            components["decision_fusion"] = {
                "status": "operational" if is_operational else "degraded",
                **dfe_health,
            }
        except Exception as exc:
            components["decision_fusion"] = {"status": "error", "error": str(exc)}

        # ── WebSocket ─────────────────────────────────────────────────────
        try:
            ws = self._state.ws_manager
            conn_count = getattr(ws, "connection_count", 0)
            components["websocket"] = {
                "status": "ok",
                "active_connections": conn_count,
            }
        except Exception as exc:
            components["websocket"] = {"status": "error", "error": str(exc)}

        # ── Overall status ────────────────────────────────────────────────
        statuses = [c.get("status", "error") for c in components.values()]
        healthy = {"ok", "operational"}
        if any(s == "error" for s in statuses):
            overall = "degraded"
        elif any(s == "degraded" for s in statuses):
            overall = "degraded"
        else:
            overall = "operational"

        return {
            "status": overall,
            "uptime_seconds": round(uptime_s, 1),
            "components": components,
        }

    async def quick_status(self) -> dict:
        """Lightweight status — running components only, no detailed checks."""
        uptime_s = time.monotonic() - self._start_time
        engines = [
            "rolling_buffer", "ml_pipeline", "scheduler",
            "forex_factory", "economic_intelligence",
            "market_intelligence_ai", "decision_fusion", "websocket",
        ]
        present = []
        for name in engines:
            attr_map = {
                "rolling_buffer": "rolling_buffer",
                "ml_pipeline": "pipeline_manager",
                "scheduler": "scheduler",
                "forex_factory": "ff_connector",
                "economic_intelligence": "eie",
                "market_intelligence_ai": "mia",
                "decision_fusion": "dfe",
                "websocket": "ws_manager",
            }
            if hasattr(self._state, attr_map[name]):
                present.append(name)

        return {
            "status": "operational",
            "uptime_seconds": round(uptime_s, 1),
            "engines_online": present,
            "engine_count": len(present),
        }
