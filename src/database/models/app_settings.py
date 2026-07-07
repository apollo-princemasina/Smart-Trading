"""ORM model for runtime-configurable application settings."""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class AppSettings(Base):
    __tablename__ = "app_settings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    # Raw string value — typed on read by SettingsService
    value: Mapped[str] = mapped_column(Text, nullable=False)

    # One of: string | bool | int | float | json
    value_type: Mapped[str] = mapped_column(String(20), nullable=False, default="string")

    description: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Grouping: inference | display | notifications | thresholds | general
    category: Mapped[str] = mapped_column(String(50), nullable=False, default="general", index=True)

    # Redact value in API responses when True
    is_secret: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<AppSettings {self.key}={self.value!r} type={self.value_type}>"
