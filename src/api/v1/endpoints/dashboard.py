"""Dashboard endpoint — single aggregated snapshot of all engine outputs."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get(
    "",
    summary="Full dashboard snapshot",
    description=(
        "Returns the current recommendation, latest prediction, market regime, "
        "MIA summary, EIE summary, buffer status, and system summary in one call."
    ),
)
async def get_dashboard(request: Request):
    svc = request.app.state.dashboard_service
    snapshot = await svc.snapshot()

    # Coerce nested objects to plain dicts for JSON serialisation
    return snapshot
