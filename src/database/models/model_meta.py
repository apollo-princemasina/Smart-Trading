"""Model metadata — records what bundle is loaded and its training provenance."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from src.database.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id:            Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    loaded_at:     Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    model_name:     Mapped[str]         = mapped_column(String(100))
    model_version:  Mapped[str | None]  = mapped_column(String(50),  nullable=True)
    bundle_path:    Mapped[str]         = mapped_column(String(512))
    feature_count:  Mapped[int]         = mapped_column(Integer, default=247)

    training_start: Mapped[str | None]  = mapped_column(String(20), nullable=True)
    training_end:   Mapped[str | None]  = mapped_column(String(20), nullable=True)
    accuracy:       Mapped[float | None] = mapped_column(Float, nullable=True)

    is_active:      Mapped[bool]        = mapped_column(Boolean, default=True, index=True)
    metadata_json:  Mapped[dict | None] = mapped_column(JSON,    nullable=True)

    def __repr__(self) -> str:
        return f"<ModelMetadata {self.model_name} active={self.is_active}>"
