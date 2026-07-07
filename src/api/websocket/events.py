"""WebSocket event type registry.

Single source of truth for all event strings used across MFIP.
Import WSEventType wherever you call ws_manager.broadcast() or
notification_service.broadcast().
"""
from __future__ import annotations


class WSEventType:
    # ── Existing (Phase 1) ────────────────────────────────────────────────
    SIGNAL_UPDATE  = "signal_update"    # new BUY/SELL/HOLD from inference
    REGIME_UPDATE  = "regime_update"    # market regime changed
    CANDLE_UPDATE  = "candle_update"    # new M15 candle in buffer
    HEALTH_UPDATE  = "health_update"    # 60-second health tick

    # ── Phase 5 — Decision Fusion ─────────────────────────────────────────
    DECISION_UPDATE = "decision_update"  # new DecisionObject produced

    # ── Phase 4 — Market Intelligence AI ─────────────────────────────────
    MIA_UPDATE = "mia_update"            # new MarketIntelligenceOutput

    # ── Phase 3 — Economic Intelligence ──────────────────────────────────
    EIE_UPDATE = "eie_update"            # active reports changed

    # ── Application Backend ───────────────────────────────────────────────
    SYSTEM_STATUS   = "system_status"    # engine state transition
    SCHEDULER_TICK  = "scheduler_tick"   # M15 cron fired
    MODEL_LOADED    = "model_loaded"     # new model bundle registered

    # ── Connection lifecycle ──────────────────────────────────────────────
    CONNECTION_ACK      = "connection_ack"      # sent on connect with current snapshot
    SUBSCRIPTION_ACK    = "subscription_ack"    # acknowledgement of subscribe message
    PING                = "ping"
    PONG                = "pong"

    # ── All event types in a set for fast membership tests ───────────────
    ALL: frozenset[str] = frozenset({
        SIGNAL_UPDATE, REGIME_UPDATE, CANDLE_UPDATE, HEALTH_UPDATE,
        DECISION_UPDATE, MIA_UPDATE, EIE_UPDATE,
        SYSTEM_STATUS, SCHEDULER_TICK, MODEL_LOADED,
        CONNECTION_ACK, SUBSCRIPTION_ACK, PING, PONG,
    })
