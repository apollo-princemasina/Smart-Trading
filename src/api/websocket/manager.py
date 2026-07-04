"""WebSocket connection manager.

Maintains a set of active connections and broadcasts messages to all of them.
The frontend connects once on dashboard load and receives real-time events
without polling.

Event types broadcast:
    signal_update  — new BUY/SELL/HOLD signal produced
    regime_update  — market regime changed
    candle_update  — new M15 candle appended to buffer
    health_update  — system health status tick (every 60s)
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("WebSocket connected — active={}", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections = [c for c in self._connections if c is not ws]
        logger.info("WebSocket disconnected — active={}", len(self._connections))

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Send a JSON message to all connected clients."""
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

        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                self._connections = [c for c in self._connections if c not in dead]
            logger.debug("Removed {} dead WebSocket connections", len(dead))

    @property
    def connection_count(self) -> int:
        return len(self._connections)
