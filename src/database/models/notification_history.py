"""ORM model for WebSocket broadcast log."""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, DateTime, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class NotificationHistory(Base):
    __tablename__ = "notification_history"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # websocket | future: email | webhook
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="websocket")

    # Mirrors WSEventType values
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # None for broadcast-to-all; set for targeted sends
    recipient: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Full JSON payload that was broadcast
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Number of clients that received it
    delivered_to: Mapped[int] = mapped_column(
        String(10), nullable=False, default=0  # stored as int via JSON numeric
    )

    delivered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Error message if delivery failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<NotificationHistory {self.event_type} channel={self.channel}>"
