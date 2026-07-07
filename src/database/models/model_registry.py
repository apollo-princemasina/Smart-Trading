"""ORM model for full ML model governance and versioning."""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class ModelRegistry(Base):
    """Comprehensive model versioning record — one row per deployed model bundle."""

    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    registered_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Identity
    model_name:    Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(50),  nullable=False)
    bundle_path:   Mapped[str] = mapped_column(String(512), nullable=False)

    # Provenance
    git_commit:            Mapped[str | None] = mapped_column(String(40),  nullable=True)
    feature_schema_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    label_version:          Mapped[str | None] = mapped_column(String(50), nullable=True)
    decision_schema_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    pipeline_version:       Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Training lineage
    training_start:   Mapped[str | None] = mapped_column(String(20),  nullable=True)
    training_end:     Mapped[str | None] = mapped_column(String(20),  nullable=True)
    training_dataset: Mapped[str | None] = mapped_column(String(200), nullable=True)
    feature_count:    Mapped[int]        = mapped_column(Integer, nullable=False, default=247)

    # Overall metrics
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Per-class metrics
    precision_buy:  Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_buy:     Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_buy:         Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_sell: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_sell:    Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_sell:        Mapped[float | None] = mapped_column(Float, nullable=True)

    # Deployment
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )

    notes:   Mapped[str | None]   = mapped_column(Text,    nullable=True)
    metrics: Mapped[dict | None]  = mapped_column(JSON,    nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry {self.model_name} v{self.model_version} "
            f"active={self.is_active}>"
        )
