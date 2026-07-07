"""DFE agreement endpoint — GET /decision/agreement."""
from __future__ import annotations

from fastapi import APIRouter, Request

from decision_fusion.api.schemas import AgreementBreakdownResponse
from decision_fusion.recommendation_cache.cache import decision_cache

router = APIRouter()


@router.get(
    "/agreement",
    response_model=AgreementBreakdownResponse,
    summary="Agreement Breakdown",
    description=(
        "Returns the current inter-source agreement and conflict scores, "
        "the consensus level, and which sources are aligned vs. conflicting."
    ),
)
async def get_agreement(request: Request):
    current = decision_cache.current

    if current is None:
        return AgreementBreakdownResponse(
            agreement_score       = None,
            conflict_score        = None,
            consensus_level       = None,
            aligned_sources       = [],
            conflicting_sources   = [],
            neutral_sources       = [],
            has_current_decision  = False,
        )

    # Recover source lists from the explanation fields
    aligned     = current.supporting_evidence[:5]   # first N supporting items
    conflicting = current.conflicting_reasons[:5]

    return AgreementBreakdownResponse(
        agreement_score       = current.agreement_score,
        conflict_score        = current.conflict_score,
        consensus_level       = current.consensus_level,
        aligned_sources       = aligned,
        conflicting_sources   = conflicting,
        neutral_sources       = [],
        has_current_decision  = True,
    )
