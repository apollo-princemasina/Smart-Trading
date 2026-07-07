"""GET /intelligence/context — Full economic context snapshot."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from economic_intelligence.intelligence_cache.cache import intelligence_cache
from economic_intelligence.api.schemas import (
    IntelligenceContextResponse,
    ExecutionContextOut,
    EconomicIntelligenceOut,
)

router = APIRouter()


def _report_to_out(r) -> EconomicIntelligenceOut:
    return EconomicIntelligenceOut(**r.__dict__)


@router.get("/context", response_model=IntelligenceContextResponse)
async def get_intelligence_context():
    """
    Full economic intelligence context snapshot.

    Returns:
    - Current execution context (risk + readiness + rationale)
    - All active events (released, still carrying influence)
    - Upcoming scheduled events (next 24 hours)
    """
    if not intelligence_cache.is_populated:
        raise HTTPException(503, detail="EIE cache not yet populated — try again shortly")

    context = await intelligence_cache.get_context()
    active = await intelligence_cache.get_active()
    upcoming = await intelligence_cache.get_upcoming(limit_hours=24.0)
    now = datetime.now(timezone.utc)

    if context is None:
        raise HTTPException(503, detail="EIE context not yet computed — try again shortly")

    context_out = ExecutionContextOut(
        execution_risk=context.execution_risk,
        execution_readiness=context.execution_readiness,
        risk_rationale=context.risk_rationale,
        readiness_rationale=context.readiness_rationale,
        time_to_next_high_min=context.time_to_next_high_min,
        active_event_count=context.active_event_count,
        upcoming_event_count=context.upcoming_event_count,
        is_market_open=context.is_market_open,
        is_holiday=context.is_holiday,
        generated_at=now,
    )

    return IntelligenceContextResponse(
        context=context_out,
        active_events=[_report_to_out(r) for r in active],
        upcoming_events=[_report_to_out(r) for r in upcoming],
        total_active=len(active),
        total_upcoming=len(upcoming),
        generated_at=now,
    )
