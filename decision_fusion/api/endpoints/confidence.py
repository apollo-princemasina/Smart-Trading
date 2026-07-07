"""DFE confidence endpoint — GET /decision/confidence."""
from __future__ import annotations

from fastapi import APIRouter, Request

from decision_fusion.api.schemas import ConfidenceBreakdownResponse
from decision_fusion.recommendation_cache.cache import decision_cache

router = APIRouter()


@router.get(
    "/confidence",
    response_model=ConfidenceBreakdownResponse,
    summary="Confidence Breakdown",
    description=(
        "Returns a detailed breakdown of how the current decision confidence was computed: "
        "per-source confidence values, agreement/conflict scores, and confidence drivers."
    ),
)
async def get_confidence(request: Request):
    current = decision_cache.current

    if current is None:
        return ConfidenceBreakdownResponse(
            decision_confidence   = None,
            ml_confidence         = None,
            eie_confidence        = None,
            ai_confidence         = None,
            agreement_score       = None,
            conflict_score        = None,
            consensus_level       = None,
            confidence_drivers    = [],
            has_current_decision  = False,
        )

    # Extract per-source confidence hints from the decision's confidence_drivers
    ml_conf  = None
    eie_conf = None
    ai_conf  = None
    for driver in current.confidence_drivers:
        if "ML model anchor" in driver and "%" in driver:
            try:
                pct = driver.split("with")[1].strip().split("%")[0].strip()
                ml_conf = round(float(pct), 1)
            except (IndexError, ValueError):
                pass

    return ConfidenceBreakdownResponse(
        decision_confidence   = current.decision_confidence,
        ml_confidence         = ml_conf,
        eie_confidence        = eie_conf,
        ai_confidence         = ai_conf,
        agreement_score       = current.agreement_score,
        conflict_score        = current.conflict_score,
        consensus_level       = current.consensus_level,
        confidence_drivers    = current.confidence_drivers,
        has_current_decision  = True,
    )
