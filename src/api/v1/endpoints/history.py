"""History endpoints — paginated prediction and decision history."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query, Request

from src.api.schemas.history import (
    CombinedHistoryItem,
    CombinedHistoryResponse,
    DecisionHistoryItem,
    DecisionHistoryResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
)

router = APIRouter(prefix="/history", tags=["History"])


@router.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
    summary="Paginated ML prediction history",
)
async def prediction_history(
    request: Request,
    symbol:    str       = Query("EURUSD"),
    direction: str | None = Query(None, description="BUY | SELL | HOLD"),
    page:      int       = Query(1, ge=1),
    page_size: int       = Query(20, ge=1, le=100),
) -> PredictionHistoryResponse:
    svc = request.app.state.history_service
    rows, total = await svc.get_prediction_history(
        symbol=symbol,
        direction=direction,
        page=page,
        page_size=page_size,
    )
    items = [
        PredictionHistoryItem(
            id=str(r.id),
            signal_time=r.signal_time,
            symbol=r.symbol,
            timeframe=r.timeframe,
            direction=r.direction,
            confidence=r.confidence,
            raw_direction=(r.metadata_json or {}).get("raw_direction") or r.direction,
            raw_confidence=r.raw_confidence,
            demoted=(r.metadata_json or {}).get("demoted", False),
            prob_buy=r.prob_buy,
            prob_sell=r.prob_sell,
            prob_hold=r.prob_hold,
            session=r.session,
            session_mult=r.session_mult,
            regime=r.regime,
            close=r.close,
            tp_price=r.tp_price,
            sl_price=r.sl_price,
            tp_pips=r.tp_pips,
            sl_pips=r.sl_pips,
            atr_pips=r.atr_pips,
        )
        for r in rows
    ]
    return PredictionHistoryResponse(
        predictions=items, total=total, page=page, page_size=page_size
    )


@router.get(
    "/decisions",
    response_model=DecisionHistoryResponse,
    summary="Paginated DFE decision history",
)
async def decision_history(
    request: Request,
    recommendation: str | None  = Query(None, description="BUY | SELL | WAIT"),
    strength:       str | None  = Query(None, description="WEAK | MODERATE | STRONG | VERY_STRONG"),
    after:          datetime | None = Query(None),
    before:         datetime | None = Query(None),
    page:           int         = Query(1, ge=1),
    page_size:      int         = Query(20, ge=1, le=100),
) -> DecisionHistoryResponse:
    svc = request.app.state.history_service
    rows, total = await svc.get_decision_history(
        recommendation=recommendation,
        strength=strength,
        after=after,
        before=before,
        page=page,
        page_size=page_size,
    )
    items = [
        DecisionHistoryItem(
            id=str(r.id),
            decision_id=r.decision_id,
            generated_at=r.generated_at,
            expires_at=r.expires_at,
            recommendation=r.recommendation,
            strength=r.strength,
            confidence=r.confidence,
            agreement_score=r.agreement_score,
            conflict_score=r.conflict_score,
            consensus_level=r.consensus_level,
            market_bias=r.market_bias,
            primary_reasons=r.primary_reasons or [],
            risk_factors=r.risk_factors or [],
            has_ml=r.has_ml,
            has_eie=r.has_eie,
            has_mia=r.has_mia,
            schema_version=r.schema_version,
        )
        for r in rows
    ]
    return DecisionHistoryResponse(
        decisions=items, total=total, page=page, page_size=page_size
    )


@router.get(
    "/combined",
    response_model=CombinedHistoryResponse,
    summary="Interleaved prediction + decision activity feed",
)
async def combined_history(
    request: Request,
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> CombinedHistoryResponse:
    svc = request.app.state.history_service
    items = await svc.get_combined_history(page=page, page_size=page_size)
    return CombinedHistoryResponse(
        items=[CombinedHistoryItem(**item) for item in items],
        page=page,
        page_size=page_size,
    )
