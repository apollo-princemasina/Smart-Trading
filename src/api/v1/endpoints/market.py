"""Market data endpoints — reads from the in-memory rolling buffer."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from src.api.core.dependencies import get_rolling_buffer, get_inference_engine
from src.api.core.exceptions    import BufferNotReadyError
from src.api.schemas.market     import CandleBufferOut, CandleOut, RegimeOut, ICTSignals, RegimeScores

router = APIRouter(prefix="/market", tags=["Market"])

_VALID_TF = {"M15", "H1", "H4", "D1", "W1"}


@router.get(
    "/candles/{timeframe}",
    response_model=CandleBufferOut,
    summary="Recent candles from in-memory rolling buffer",
)
async def get_candles(
    timeframe: str = Path(..., description="M15 | H1 | H4 | D1 | W1"),
    limit:     int = Query(100, ge=1, le=800),
    buffer=Depends(get_rolling_buffer),
):
    tf = timeframe.upper()
    if tf not in _VALID_TF:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid timeframe. Choose from: {sorted(_VALID_TF)}",
        )
    if not buffer.is_ready(tf):
        raise BufferNotReadyError(f"Buffer for {tf} is not yet populated")

    candles = buffer.get_candles(tf, limit=limit)
    return CandleBufferOut(
        symbol="EURUSD",
        timeframe=tf,
        count=len(candles),
        candles=[
            CandleOut(
                timestamp=c["timestamp"],
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=int(c.get("tick_volume", 0)),
            )
            for c in candles
        ],
    )


@router.get(
    "/regime",
    response_model=RegimeOut,
    summary="Current market regime (Consolidation / Expansion / Manipulation)",
)
async def get_regime(
    engine=Depends(get_inference_engine),
    buffer=Depends(get_rolling_buffer),
):
    if not buffer.is_ready("M15"):
        raise BufferNotReadyError("Buffer not ready — cannot compute regime")

    regime = engine.latest_regime()
    if not regime:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No regime computed yet — wait for first M15 bar close",
        )

    scores = regime.regime_scores
    return RegimeOut(
        symbol="EURUSD",
        timeframe="M15",
        timestamp=regime.timestamp,
        dominant=regime.dominant_regime,
        scores=RegimeScores(
            consolidation=scores.get("CONSOLIDATION", 0.0),
            expansion=scores.get("EXPANSION", 0.0),
            manipulation=scores.get("MANIPULATION", 0.0),
        ),
        bias=regime.bias,
        pd_zone=regime.pd_zone,
        atr_pips=regime.atr_pips,
        atr_vs_avg=regime.atr_vs_avg,
        adx=regime.adx,
        ict=ICTSignals(
            liquidity_sweep=regime.liquidity_sweep,
            sweep_direction=regime.sweep_direction,
            sweep_rejected=regime.sweep_rejected,
            sweep_confirmed=regime.sweep_confirmed,
            choch_detected=regime.choch_detected,
            choch_direction=regime.choch_direction,
            bos_detected=regime.bos_detected,
            bos_direction=regime.bos_direction,
            fvg_active=regime.fvg_active,
            fvg_direction=regime.fvg_direction,
            ob_active=regime.ob_active,
            ob_direction=regime.ob_direction,
            in_order_block=regime.in_order_block,
        ),
        narrative=regime.narrative,
        trade_implication=regime.trade_implication,
    )
