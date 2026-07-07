"""DFE health endpoint — GET /decision/health."""
from __future__ import annotations

from fastapi import APIRouter, Request

from decision_fusion.api.schemas import DFEHealthResponse
from decision_fusion.recommendation_cache.cache import decision_cache

router = APIRouter()


@router.get(
    "/health",
    response_model=DFEHealthResponse,
    summary="DFE Health",
    description=(
        "Returns the Decision Fusion Engine health dashboard: "
        "current recommendation, expiry, agreement scores, processing time, and cache status."
    ),
)
async def get_dfe_health(request: Request):
    dfe = getattr(request.app.state, "dfe", None)

    if dfe is None:
        return DFEHealthResponse(
            status                   = "offline",
            running                  = False,
            schema_version           = "decision_fusion_v1",
            current_recommendation   = None,
            recommendation_strength  = None,
            recommendation_age_s     = None,
            time_until_expiry_s      = None,
            is_expired               = True,
            agreement_score          = None,
            conflict_score           = None,
            decision_confidence      = None,
            avg_processing_ms        = None,
            total_decisions          = 0,
            cache_size               = 0,
        )

    h = dfe.health()
    return DFEHealthResponse(
        status                   = h.get("status", "offline"),
        running                  = h.get("running", False),
        schema_version           = h.get("schema_version", "decision_fusion_v1"),
        current_recommendation   = h.get("current_recommendation"),
        recommendation_strength  = h.get("recommendation_strength"),
        recommendation_age_s     = h.get("recommendation_age_s"),
        time_until_expiry_s      = h.get("time_until_expiry_s"),
        is_expired               = h.get("is_expired", True),
        agreement_score          = h.get("agreement_score"),
        conflict_score           = h.get("conflict_score"),
        decision_confidence      = h.get("decision_confidence"),
        avg_processing_ms        = h.get("avg_processing_ms"),
        total_decisions          = h.get("total_decisions", 0),
        cache_size               = h.get("cache_size", 0),
    )
