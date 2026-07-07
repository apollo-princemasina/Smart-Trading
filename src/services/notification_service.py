"""NotificationService — WebSocket broadcast with DB logging."""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.database.models.notification_history import NotificationHistory


class NotificationService:
    """
    Thin wrapper around WebSocketManager that persists every broadcast
    to notification_history for audit and replay.
    """

    def __init__(self, ws_manager, session_factory) -> None:
        self._ws = ws_manager
        self._session_factory = session_factory

    async def broadcast(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        persist: bool = True,
    ) -> int:
        """
        Broadcast to all connected WebSocket clients.

        Returns the number of active connections at broadcast time.
        """
        conn_count = getattr(self._ws, "connection_count", 0)

        try:
            await self._ws.broadcast(event_type, data)
            delivered = True
            error = None
        except Exception as exc:
            delivered = False
            error = str(exc)
            logger.error("WebSocket broadcast failed for {}: {}", event_type, exc)

        if persist:
            await self._persist(
                event_type=event_type,
                payload={"event": event_type, "data": data},
                delivered=delivered,
                delivered_to=conn_count,
                error=error,
            )

        return conn_count

    async def get_history(self, limit: int = 50) -> list[NotificationHistory]:
        async with self._session_factory() as session:
            from src.database.repositories.base import BaseRepository
            from sqlalchemy import select
            result = await session.execute(
                select(NotificationHistory)
                .order_by(NotificationHistory.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # ── Private ───────────────────────────────────────────────────────────

    async def _persist(
        self,
        *,
        event_type: str,
        payload: dict,
        delivered: bool,
        delivered_to: int,
        error: str | None,
    ) -> None:
        try:
            async with self._session_factory() as session:
                entry = NotificationHistory(
                    event_type=event_type,
                    payload=payload,
                    delivered=delivered,
                    delivered_to=delivered_to,
                    error=error,
                )
                session.add(entry)
                await session.commit()
        except Exception as exc:
            logger.warning("Could not persist notification log: {}", exc)
