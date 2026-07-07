"""DecisionService — wraps DFE with DB persistence."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from src.database.models.decision_history import DecisionHistory
from src.database.repositories.decision_repo import DecisionRepository


class DecisionService:
    """
    Orchestrates the Decision Fusion Engine and ensures every produced
    DecisionObject is persisted to decision_history for analytics.

    Does NOT contain fusion logic — that lives entirely in the DFE.
    """

    def __init__(self, session_factory, app_state) -> None:
        self._session_factory = session_factory
        self._state = app_state

    # ── Current decision ──────────────────────────────────────────────────

    def current(self) -> Any | None:
        """Return the live DecisionObject from the DFE in-memory cache."""
        try:
            from decision_fusion.recommendation_cache.cache import decision_cache
            return decision_cache.current
        except Exception:
            return None

    def cache_stats(self) -> dict:
        try:
            from decision_fusion.recommendation_cache.cache import decision_cache
            return decision_cache.stats()
        except Exception:
            return {}

    # ── History ───────────────────────────────────────────────────────────

    async def get_history(
        self,
        *,
        recommendation: str | None = None,
        strength: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[DecisionHistory], int]:
        async with self._session_factory() as session:
            repo = DecisionRepository(session)
            rows, total = await repo.list_paginated(
                recommendation=recommendation,
                strength=strength,
                after=after,
                before=before,
                page=page,
                page_size=page_size,
            )
            return list(rows), total

    # ── Persistence ───────────────────────────────────────────────────────

    async def persist(self, decision: Any) -> bool:
        """
        Persist a DecisionObject to the decision_history table.

        Returns True if saved, False if already exists (idempotent).
        """
        if decision is None:
            return False

        decision_id = getattr(decision, "decision_id", None)
        if not decision_id:
            return False

        try:
            async with self._session_factory() as session:
                repo = DecisionRepository(session)
                if await repo.exists(decision_id):
                    return False

                row = DecisionHistory(
                    decision_id=decision_id,
                    generated_at=getattr(decision, "generated_at", None),
                    expires_at=getattr(decision, "expires_at", None),
                    schema_version=getattr(decision, "decision_schema_version", "decision_fusion_v1"),
                    recommendation=str(getattr(decision, "recommendation", "")).replace("Recommendation.", ""),
                    strength=str(getattr(decision, "recommendation_strength", "")).replace("RecommendationStrength.", ""),
                    confidence=float(getattr(decision, "decision_confidence", 0.0)),
                    agreement_score=float(getattr(decision, "agreement_score", 0.0)),
                    conflict_score=float(getattr(decision, "conflict_score", 0.0)),
                    consensus_level=str(getattr(decision, "consensus_level", "")).replace("ConsensusLevel.", ""),
                    technical_alignment=float(getattr(decision, "technical_alignment", 0.0)),
                    fundamental_alignment=float(getattr(decision, "fundamental_alignment", 0.0)),
                    market_bias=str(getattr(decision, "market_bias", "")).replace("MarketBiasEnum.", ""),
                    primary_reasons=list(getattr(decision, "primary_reasons", [])),
                    supporting_evidence=list(getattr(decision, "supporting_evidence", [])),
                    conflicting_reasons=list(getattr(decision, "conflicting_reasons", [])),
                    confidence_drivers=list(getattr(decision, "confidence_drivers", [])),
                    risk_factors=list(getattr(decision, "risk_factors", [])),
                    has_ml=bool(getattr(decision, "has_ml", False)),
                    has_eie=bool(getattr(decision, "has_eie", False)),
                    has_mia=bool(getattr(decision, "has_mia", False)),
                )
                session.add(row)
                await session.commit()
                logger.debug("DecisionHistory saved: {} {}", row.recommendation, decision_id[:8])
                return True
        except Exception as exc:
            logger.warning("Could not persist decision {}: {}", decision_id, exc)
            return False
