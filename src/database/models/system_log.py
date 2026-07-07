"""ORM model for structured system event logs."""
from __future__ import annotations

import uuid

from sqlalchemy import DateTime, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class SystemLog(Base):
    __tablename__ = "system_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    logged_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # DEBUG | INFO | WARNING | ERROR | CRITICAL
    level: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Which engine / module emitted this event
    component: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Structured event identifier (e.g. "dfe_started", "buffer_refill_failed")
    event_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)

    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional structured payload (arbitrary JSON)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Request-ID or cycle-ID for cross-component correlation
    correlation_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )

    def __repr__(self) -> str:
        return f"<SystemLog [{self.level}] {self.component}/{self.event_type}>"
