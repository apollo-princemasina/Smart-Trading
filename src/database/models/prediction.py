"""Prediction ORM model — one row per M15 bar inference result."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from src.database.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    signal_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    symbol:    Mapped[str] = mapped_column(String(10), default="EURUSD")
    timeframe: Mapped[str] = mapped_column(String(5),  default="M15")

    # Signal
    direction:  Mapped[str]   = mapped_column(String(4))   # BUY | SELL | HOLD
    confidence: Mapped[float] = mapped_column(Float)
    prob_sell:  Mapped[float] = mapped_column(Float)
    prob_hold:  Mapped[float] = mapped_column(Float)
    prob_buy:   Mapped[float] = mapped_column(Float)

    # Price context at signal time
    close:    Mapped[float]        = mapped_column(Float)
    atr_pips: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp_pips:  Mapped[float | None] = mapped_column(Float, nullable=True)
    sl_pips:  Mapped[float | None] = mapped_column(Float, nullable=True)

    # Regime at signal time
    regime:        Mapped[str | None] = mapped_column(String(20),  nullable=True)
    regime_scores: Mapped[dict | None] = mapped_column(JSON,       nullable=True)

    # Session weighting
    raw_confidence: Mapped[float | None] = mapped_column(Float,      nullable=True)
    session:        Mapped[str | None]   = mapped_column(String(20), nullable=True)
    session_mult:   Mapped[float | None] = mapped_column(Float,      nullable=True)

    # Model provenance
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON,      nullable=True)

    # Relationship
    outcome: Mapped["PredictionOutcome | None"] = relationship(
        "PredictionOutcome", back_populates="prediction", uselist=False, lazy="select"
    )

    __table_args__ = (
        Index("ix_predictions_symbol_timeframe", "symbol", "timeframe"),
        Index("ix_predictions_direction", "direction"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id[:8]} "
            f"{self.symbol} {self.timeframe} "
            f"{self.direction} {self.confidence:.0%} "
            f"@ {self.signal_time}>"
        )
