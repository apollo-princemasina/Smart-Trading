"""
MarketIntelligenceAIEngine — top-level orchestrator for the MIA layer.

Lifecycle: startup() → background APScheduler cycle → shutdown()

The engine:
  1. Holds a single AIGateway connected to the configured provider
  2. Exposes a single MarketIntelligenceAgent
  3. Runs a background cycle to re-analyse high-impact events
  4. Maintains an in-memory store of the last N analyses for the API
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from market_intelligence_ai.agent.market_agent import MarketIntelligenceAgent
from market_intelligence_ai.ai_cache.cache import ai_cache
from market_intelligence_ai.ai_gateway.gateway import AIGateway
from market_intelligence_ai.market_context_compiler.compiler import MarketContextCompiler
from market_intelligence_ai.market_context_compiler.context_models import (
    ContextPayload,
    EventTrigger,
    EIESnapshot,
)
from market_intelligence_ai.providers.groq_provider import GroqProvider
from market_intelligence_ai.schema.market_intelligence import MarketIntelligenceOutput
from market_intelligence_ai.utils.config import mia_config
from market_intelligence_ai.utils.logger import logger

_MAX_STORED_ANALYSES = 200
_GENERAL_NARRATIVE_INTERVAL_S = 1800  # run general market narrative every 30 min


class MarketIntelligenceAIEngine:
    """
    Main lifecycle manager for the Market Intelligence AI Layer.

    Usage:
        engine = MarketIntelligenceAIEngine()
        await engine.startup()
        analysis = await engine.agent.analyze(payload)
        await engine.shutdown()
    """

    def __init__(self) -> None:
        self._provider          = GroqProvider(api_key=mia_config.GROQ_API_KEY)
        self._gateway           = AIGateway(self._provider)
        self._context_compiler  = MarketContextCompiler()
        self._agent             = MarketIntelligenceAgent(
            gateway          = self._gateway,
            context_compiler = self._context_compiler,
            cache            = ai_cache,
        )
        self._scheduler   = AsyncIOScheduler()
        self._analyses:   deque[MarketIntelligenceOutput] = deque(maxlen=_MAX_STORED_ANALYSES)
        self._lock        = asyncio.Lock()
        self._running     = False
        self._last_general_at: Optional[datetime] = None
        self._market_ctx_provider = None  # injected by main.py after all services start

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def agent(self) -> MarketIntelligenceAgent:
        return self._agent

    @property
    def gateway(self) -> AIGateway:
        return self._gateway

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_analysis(self) -> Optional[MarketIntelligenceOutput]:
        """Return the most recently stored analysis, or None."""
        return self._analyses[-1] if self._analyses else None

    def set_market_context_provider(self, provider_fn) -> None:
        """Register a zero-arg callable that returns a dict of live market context."""
        self._market_ctx_provider = provider_fn

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        logger.info("MIA engine starting up…")
        if not mia_config.groq_configured:
            logger.warning(
                "MIA: GROQ_API_KEY is not set — analysis requests will return fallbacks. "
                "Set GROQ_API_KEY in .env to enable live AI analysis."
            )

        self._scheduler.add_job(
            self._cycle,
            trigger          = "interval",
            seconds          = mia_config.MIA_CYCLE_SECONDS,
            id               = "mia_cycle",
            replace_existing = True,
            next_run_time    = datetime.now(timezone.utc),  # fire immediately on startup
        )
        self._scheduler.start()
        self._running = True
        logger.info("MIA engine started (cycle every {}s).", mia_config.MIA_CYCLE_SECONDS)

    async def shutdown(self) -> None:
        logger.info("MIA engine shutting down…")
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("MIA engine stopped.")

    # ── Background cycle ──────────────────────────────────────────────────────

    async def _cycle(self) -> None:
        """
        Background cycle: re-analyse active high-impact events from the EIE.

        Pulls the current EIE context and re-submits any new or stale high-impact
        events that have not been analysed recently.
        """
        logger.debug("MIA cycle starting…")
        try:
            eie_snapshot = await self._build_eie_snapshot()
            processed = 0

            for event_dict in eie_snapshot.active_events:
                if event_dict.get("impact_level") not in ("HIGH",):
                    continue  # only auto-analyse HIGH-impact active events

                event_id = event_dict.get("event_id")
                if not event_id:
                    continue

                try:
                    trigger = EventTrigger(
                        event_id           = event_id,
                        title              = event_dict.get("title", "Unknown Event"),
                        currency           = event_dict.get("currency", "USD"),
                        timestamp          = datetime.now(timezone.utc),
                        importance         = event_dict.get("impact_level", "MEDIUM"),
                        actual             = event_dict.get("actual"),
                        forecast           = event_dict.get("forecast"),
                        previous           = event_dict.get("previous"),
                        surprise_class     = event_dict.get("surprise_class", "NONE"),
                        surprise_direction = event_dict.get("surprise_direction", "IN_LINE"),
                        economic_direction = event_dict.get("economic_direction", "UNCERTAIN"),
                    )
                    payload = self._context_compiler.build_for_event(trigger, eie_snapshot)
                    result  = await self._agent.analyze(payload)
                    await self._store(result)
                    processed += 1

                except Exception as exc:
                    logger.error("MIA cycle: failed to analyse event {} — {}", event_id, exc)

            logger.debug("MIA cycle: processed {} high-impact events.", processed)

            # General market narrative — runs every 30 min regardless of active events
            now = datetime.now(timezone.utc)
            general_due = (
                self._last_general_at is None
                or (now - self._last_general_at).total_seconds() >= _GENERAL_NARRATIVE_INTERVAL_S
            )
            if general_due:
                try:
                    market_ctx: dict = {}
                    if self._market_ctx_provider:
                        try:
                            market_ctx = self._market_ctx_provider() or {}
                        except Exception:
                            pass
                    payload = self._context_compiler.build_for_general_context(
                        eie_snapshot, market_ctx=market_ctx
                    )
                    result  = await self._agent.analyze(payload)
                    await self._store(result)
                    self._last_general_at = now
                    logger.info(
                        "MIA: general market narrative generated (session={}, bias={}).",
                        payload.current_session, result.market_bias,
                    )
                except Exception as exc:
                    logger.error("MIA general narrative error: {}", exc)

        except Exception as exc:
            logger.error("MIA cycle error: {}", exc)

    async def _build_eie_snapshot(self) -> EIESnapshot:
        """Pull current intelligence from the EIE cache."""
        try:
            from economic_intelligence.intelligence_cache.cache import intelligence_cache
            active_reports   = await intelligence_cache.get_active()
            upcoming_reports = await intelligence_cache.get_high_impact_upcoming(window_minutes=120)

            active_events = [
                {
                    "event_id":           r.event_id,
                    "title":              r.event_title,
                    "currency":           r.event_currency,
                    "impact_level":       r.impact_level.value if hasattr(r.impact_level, "value") else r.impact_level,
                    "economic_direction": r.economic_direction.value if hasattr(r.economic_direction, "value") else r.economic_direction,
                    "impact_score":       r.impact_score,
                    "remaining_influence": r.remaining_influence,
                    "surprise_class":     r.surprise_class.value if r.surprise_class and hasattr(r.surprise_class, "value") else "NONE",
                    "surprise_direction": r.surprise_direction.value if r.surprise_direction and hasattr(r.surprise_direction, "value") else "IN_LINE",
                    "actual":             r.actual,
                    "forecast":           r.forecast,
                    "previous":           r.previous,
                }
                for r in active_reports
            ]

            upcoming_events = [
                {
                    "event_id":   r.event_id,
                    "title":      r.event_title,
                    "currency":   r.event_currency,
                    "event_time": r.event_time.strftime("%H:%M UTC") if r.event_time else "?",
                }
                for r in upcoming_reports
            ]

            # Build dominant directions from active events
            dominant: dict[str, str] = {}
            for r in active_reports:
                ccy  = r.event_currency
                bias = r.economic_direction.value if hasattr(r.economic_direction, "value") else str(r.economic_direction)
                if ccy not in dominant:
                    dominant[ccy] = bias

            return EIESnapshot(
                dominant_directions  = dominant,
                active_events        = active_events,
                upcoming_high_impact = upcoming_events,
                execution_risk       = getattr(active_reports[0], "execution_risk", 0.0) if active_reports else 0.0,
                execution_readiness  = getattr(active_reports[0], "execution_readiness", 0.0) if active_reports else 0.0,
                snapshot_at          = datetime.now(timezone.utc),
            )

        except ImportError:
            logger.warning("MIA: EIE not available — using empty snapshot.")
            return EIESnapshot(snapshot_at=datetime.now(timezone.utc))

    async def _store(self, result: MarketIntelligenceOutput) -> None:
        async with self._lock:
            self._analyses.append(result)

    # ── API helpers ───────────────────────────────────────────────────────────

    async def get_recent_analyses(self, limit: int = 20) -> list[MarketIntelligenceOutput]:
        async with self._lock:
            items = list(self._analyses)
        return sorted(items, key=lambda x: x.timestamp, reverse=True)[:limit]

    def health(self) -> dict:
        return {
            "running":         self._running,
            "provider":        self._provider.provider_name,
            "model":           mia_config.MIA_ANALYSIS_MODEL,
            "groq_configured": mia_config.groq_configured,
            "circuit_state":   self._gateway.circuit_state,
            "cache_stats":     ai_cache.stats(),
            "gateway_metrics": self._gateway.metrics.snapshot(),
            "analyses_stored": len(self._analyses),
        }
