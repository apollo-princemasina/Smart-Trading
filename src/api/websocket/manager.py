"""WebSocket connection manager.

Maintains a set of active connections and broadcasts messages to all of them.
The frontend connects once on dashboard load and receives real-time events
without polling.

Subscription filtering (new):
    After connecting, a client can send:
        {"type": "subscribe", "events": ["decision_update", "signal_update"]}
    to receive only those event types.  Clients that never send a subscribe
    message continue to receive all events — fully backwards-compatible.

Event types:
    See src/api/websocket/events.py for the full registry.

Legacy events (still broadcast):
    signal_update  — new BUY/SELL/HOLD signal produced
    regime_update  — market regime changed
    candle_update  — new M15 candle appended to buffer
    health_update  — system health status tick (every 60s)

New events:
    decision_update — new DecisionObject from DFE
    mia_update      — new MIA output
    eie_update      — EIE active reports changed
    system_status   — engine state transition
    scheduler_tick  — M15 cron fired
    connection_ack  — snapshot sent on connect
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket
from loguru import logger


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        # Maps WebSocket → frozenset of subscribed event types, or None = all
        self._subscriptions: dict[int, frozenset[str] | None] = {}
        self._lock = asyncio.Lock()

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
            self._subscriptions[id(ws)] = None   # None = receive all events
        logger.info("WebSocket connected — active={}", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections = [c for c in self._connections if c is not ws]
            self._subscriptions.pop(id(ws), None)
        logger.info("WebSocket disconnected — active={}", len(self._connections))

    # ── Subscription management ───────────────────────────────────────────

    async def handle_message(self, ws: WebSocket, raw: str) -> None:
        """
        Process a text message from a connected client.

        Supported messages:
            {"type": "subscribe", "events": ["event_type", ...]}
            {"type": "unsubscribe"}   — go back to receiving all events
            {"type": "ping"}          — responds with pong
        """
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        msg_type = msg.get("type")

        if msg_type == "subscribe":
            from src.api.websocket.events import WSEventType
            requested = set(msg.get("events") or [])
            valid = requested & WSEventType.ALL
            async with self._lock:
                self._subscriptions[id(ws)] = frozenset(valid) if valid else None
            ack = json.dumps({
                "event":     "subscription_ack",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data":      {"subscribed": sorted(valid) if valid else "all"},
            })
            try:
                await ws.send_text(ack)
            except Exception:
                pass

        elif msg_type == "unsubscribe":
            async with self._lock:
                self._subscriptions[id(ws)] = None

        elif msg_type == "ping":
            pong = json.dumps({
                "event":     "pong",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data":      {},
            })
            try:
                await ws.send_text(pong)
            except Exception:
                pass

    # ── Broadcasting ──────────────────────────────────────────────────────

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Send a JSON message to all clients subscribed to this event type."""
        if not self._connections:
            return

        payload = json.dumps({
            "event":     event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data":      data,
        }, default=str)

        dead: list[WebSocket] = []
        async with self._lock:
            targets = list(self._connections)
            subs    = dict(self._subscriptions)

        for ws in targets:
            sub = subs.get(id(ws))
            # None means all events; frozenset means filtered
            if sub is not None and event_type not in sub:
                continue
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                self._connections = [c for c in self._connections if c not in dead]
                for d in dead:
                    self._subscriptions.pop(id(d), None)
            logger.debug("Removed {} dead WebSocket connections", len(dead))

    async def send_to(self, ws: WebSocket, event_type: str, data: dict[str, Any]) -> None:
        """Send a targeted message to a single connection."""
        payload = json.dumps({
            "event":     event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data":      data,
        }, default=str)
        try:
            await ws.send_text(payload)
        except Exception:
            await self.disconnect(ws)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    def subscription_summary(self) -> dict:
        total = len(self._connections)
        all_events  = sum(1 for v in self._subscriptions.values() if v is None)
        filtered    = total - all_events
        return {"total": total, "receive_all": all_events, "filtered": filtered}
