"""DFE decision endpoints — GET /decision/current and GET /decision/history."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from decision_fusion.api.schemas import (
    DecisionHistoryResponse,
    DecisionOut,
    DecisionResponse,
)
from decision_fusion.recommendation_cache.cache import decision_cache

router = APIRouter()


@router.get(
    "/current",
    response_model=DecisionResponse,
    summary="Current Decision",
    description=(
        "Returns the current active decision from the Decision Fusion Engine. "
        "If no decision has been generated yet, `decision` will be null. "
        "Check `is_expired` to know whether the decision should be refreshed."
    ),
)
async def get_current_decision(request: Request):
    current = decision_cache.current

    if current is None:
        return DecisionResponse(
            decision              = None,
            is_expired            = True,
            age_seconds           = None,
            seconds_until_expiry  = None,
        )

    return DecisionResponse(
        decision              = DecisionOut(**current.model_dump()),
        is_expired            = decision_cache.is_expired(),
        age_seconds           = decision_cache.age_seconds(),
        seconds_until_expiry  = decision_cache.seconds_until_expiry(),
    )


@router.get(
    "/history",
    response_model=DecisionHistoryResponse,
    summary="Decision History",
    description="Returns the last N decisions produced by the DFE (newest first).",
)
async def get_decision_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100, description="Number of decisions to return"),
):
    decisions = decision_cache.get_history(limit)
    return DecisionHistoryResponse(
        decisions=[DecisionOut(**d.model_dump()) for d in decisions],
        total=len(decisions),
    )
