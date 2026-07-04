"""Health endpoints — used by Railway, Docker, and the frontend status bar."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from src.api.core.config      import settings
from src.api.core.dependencies import get_health_monitor
from src.api.schemas.health    import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])

_START_TIME = time.monotonic()


@router.get("/", response_model=HealthResponse, summary="Full system health report")
async def health(
    request: Request,
    monitor=Depends(get_health_monitor),
) -> HealthResponse:
    return await monitor.check()


@router.get("/live", summary="Kubernetes liveness probe — is the process alive?")
async def liveness():
    """Returns 200 as long as the process is running."""
    return {"status": "alive"}


@router.get("/ready", summary="Kubernetes readiness probe — is the system ready to serve?")
async def readiness(
    request: Request,
    monitor=Depends(get_health_monitor),
):
    """Returns 200 only when the buffer is populated and the model is loaded."""
    report = await monitor.check()
    if report.status == "unhealthy":
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not ready",
        )
    return {"status": report.status}
