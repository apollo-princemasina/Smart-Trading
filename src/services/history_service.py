"""HistoryService — unified paginated history across predictions and decisions."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.database.repositories.decision_repo import DecisionRepository
from src.database.repositories.prediction_repo import PredictionRepository


class HistoryService:
    """
    Unified history layer. Provides separate and combined history views
    over predictions (Phase 1) and decisions (Phase 5).
    """

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def get_prediction_history(
        self,
        *,
        symbol: str = "EURUSD",
        direction: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list, int]:
        async with self._session_factory() as session:
            repo = PredictionRepository(session)
            rows, total = await repo.list_recent(
                symbol=symbol,
                direction=direction,
                page=page,
                page_size=page_size,
            )
            return list(rows), total

    async def get_decision_history(
        self,
        *,
        recommendation: str | None = None,
        strength: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list, int]:
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

    async def get_combined_history(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> list[dict]:
        """
        Interleave prediction and decision records sorted by timestamp descending.

        Returns dicts with a discriminator field (record_type: prediction|decision).
        """
        async with self._session_factory() as session:
            pred_repo = DecisionRepository(session)
            dec_repo  = DecisionRepository(session)

            # Fetch enough records from each to fill one page
            fetch_limit = page_size * page

            from sqlalchemy import select
            from src.database.models.prediction import Prediction
            from src.database.models.decision_history import DecisionHistory

            preds = (await session.execute(
                select(Prediction)
                .order_by(Prediction.signal_time.desc())
                .limit(fetch_limit)
            )).scalars().all()

            decs = (await session.execute(
                select(DecisionHistory)
                .order_by(DecisionHistory.generated_at.desc())
                .limit(fetch_limit)
            )).scalars().all()

        combined: list[dict[str, Any]] = []

        for p in preds:
            combined.append({
                "record_type": "prediction",
                "timestamp": p.signal_time,
                "id": str(p.id),
                "direction": p.direction,
                "confidence": p.confidence,
                "regime": p.regime,
                "symbol": p.symbol,
            })

        for d in decs:
            combined.append({
                "record_type": "decision",
                "timestamp": d.generated_at,
                "id": str(d.id),
                "recommendation": d.recommendation,
                "strength": d.strength,
                "confidence": d.confidence,
                "agreement_score": d.agreement_score,
            })

        combined.sort(key=lambda x: x["timestamp"] or datetime.min, reverse=True)

        start = (page - 1) * page_size
        return combined[start : start + page_size]
