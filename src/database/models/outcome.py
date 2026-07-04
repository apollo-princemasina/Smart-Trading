"""Prediction outcome — evaluated after TP or SL is reached."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionOutcome(Base):
    __tablename__ = "prediction_outcomes"

    id:            Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction_id: Mapped[str]      = mapped_column(String(36), ForeignKey("predictions.id"), unique=True, index=True)
    evaluated_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # TP_HIT | SL_HIT | EXPIRED | PENDING
    outcome:      Mapped[str]         = mapped_column(String(10), default="PENDING", index=True)
    exit_price:   Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pips:     Mapped[float | None] = mapped_column(Float, nullable=True)
    bars_to_exit: Mapped[int | None]   = mapped_column(Integer, nullable=True)
    notes:        Mapped[str | None]   = mapped_column(String(255), nullable=True)

    prediction: Mapped["Prediction"] = relationship(
        "Prediction", back_populates="outcome"
    )

    def __repr__(self) -> str:
        return f"<Outcome pred={self.prediction_id[:8]} {self.outcome} pnl={self.pnl_pips}>"
