"""ORM model for persisted DFE DecisionObject snapshots."""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, DateTime, Float, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class DecisionHistory(Base):
    __tablename__ = "decision_history"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Unique DFE-generated UUID — matches DecisionObject.decision_id
    decision_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)

    generated_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    expires_at:   Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False)
    schema_version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="decision_fusion_v1"
    )

    # Core recommendation
    recommendation: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    strength:       Mapped[str] = mapped_column(String(20), nullable=False)
    confidence:     Mapped[float] = mapped_column(Float, nullable=False)

    # Agreement
    agreement_score:  Mapped[float] = mapped_column(Float, nullable=False)
    conflict_score:   Mapped[float] = mapped_column(Float, nullable=False)
    consensus_level:  Mapped[str]   = mapped_column(String(20), nullable=False)

    # Alignment signals
    technical_alignment:   Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fundamental_alignment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    market_bias:           Mapped[str]   = mapped_column(String(20), nullable=False)

    # Explanation lists (stored as JSON arrays of strings)
    primary_reasons:     Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    supporting_evidence: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    conflicting_reasons: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    confidence_drivers:  Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    risk_factors:        Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Source flags
    has_ml:  Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_eie: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_mia: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return (
            f"<DecisionHistory {self.recommendation} "
            f"strength={self.strength} conf={self.confidence:.1f}>"
        )
