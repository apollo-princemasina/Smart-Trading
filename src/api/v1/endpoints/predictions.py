"""Prediction endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.core.dependencies       import get_db, get_prediction_service
from src.api.core.exceptions          import PredictionNotFoundError
from src.api.schemas.prediction       import PredictionListOut, PredictionOut, OutcomeOut
from src.database.repositories.prediction_repo import PredictionRepository

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/latest", response_model=PredictionOut, summary="Latest signal for EURUSD M15")
async def get_latest(
    symbol: str = Query("EURUSD"),
    db: AsyncSession = Depends(get_db),
):
    repo = PredictionRepository(db)
    pred = await repo.latest(symbol)
    if not pred:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No predictions yet")
    return pred


@router.get("/", response_model=PredictionListOut, summary="Paginated prediction history")
async def list_predictions(
    symbol:    str           = Query("EURUSD"),
    direction: str | None    = Query(None, description="BUY | SELL | HOLD"),
    page:      int           = Query(1, ge=1),
    page_size: int           = Query(20, ge=1, le=100),
    db:        AsyncSession  = Depends(get_db),
):
    repo = PredictionRepository(db)
    rows, total = await repo.list_recent(
        symbol=symbol, direction=direction, page=page, page_size=page_size
    )
    return PredictionListOut(
        total=total,
        page=page,
        page_size=page_size,
        predictions=rows,
    )


@router.get("/{prediction_id}", response_model=PredictionOut, summary="Single prediction by ID")
async def get_prediction(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
):
    repo = PredictionRepository(db)
    pred = await repo.get(prediction_id)
    if not pred:
        raise PredictionNotFoundError(f"Prediction {prediction_id} not found")
    return pred
