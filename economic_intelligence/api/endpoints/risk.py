"""GET /intelligence/execution-risk and /intelligence/readiness endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from economic_intelligence.intelligence_cache.cache import intelligence_cache

router = APIRouter()


class RiskResponse(BaseModel):
    execution_risk:  float
    rationale:       str
    is_market_open:  bool
    is_holiday:      bool
    generated_at:    datetime


class ReadinessResponse(BaseModel):
    execution_readiness: float
    rationale:           str
    is_market_open:      bool
    is_holiday:          bool
    generated_at:        datetime


@router.get("/execution-risk", response_model=RiskResponse)
async def get_execution_risk():
    """
    Current execution risk score (0–100).

    100 = extremely dangerous (event imminent, do not trade).
    0   = clean window, no upcoming events.
    """
    ctx = await intelligence_cache.get_context()
    if ctx is None:
        raise HTTPException(503, detail="EIE context not yet available")

    return RiskResponse(
        execution_risk=ctx.execution_risk,
        rationale=ctx.risk_rationale,
        is_market_open=ctx.is_market_open,
        is_holiday=ctx.is_holiday,
        generated_at=datetime.now(timezone.utc),
    )


@router.get("/readiness", response_model=ReadinessResponse)
async def get_execution_readiness():
    """
    Current execution readiness score (0–100).

    100 = optimal conditions for trade entry.
    0   = avoid trading (holiday or extreme risk).
    """
    ctx = await intelligence_cache.get_context()
    if ctx is None:
        raise HTTPException(503, detail="EIE context not yet available")

    return ReadinessResponse(
        execution_readiness=ctx.execution_readiness,
        rationale=ctx.readiness_rationale,
        is_market_open=ctx.is_market_open,
        is_holiday=ctx.is_holiday,
        generated_at=datetime.now(timezone.utc),
    )
