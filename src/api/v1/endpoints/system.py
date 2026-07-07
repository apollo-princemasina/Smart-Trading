"""System endpoints — health, status, version, logs."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from src.api.core.config import settings
from src.api.schemas.system import (
    SystemHealthResponse,
    SystemLogsResponse,
    SystemStatusResponse,
    SystemVersionResponse,
    VersionInfo,
)

router = APIRouter(prefix="/system", tags=["System"])


@router.get(
    "/health",
    response_model=SystemHealthResponse,
    summary="Deep health check",
    description="Checks all Phase 1–5 engines, WebSocket, scheduler, and database.",
)
async def full_health(request: Request) -> SystemHealthResponse:
    svc = request.app.state.system_health_service
    result = await svc.full_health()
    return SystemHealthResponse(**result)


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Quick operational status",
)
async def quick_status(request: Request) -> SystemStatusResponse:
    svc = request.app.state.system_health_service
    result = await svc.quick_status()
    return SystemStatusResponse(**result)


@router.get(
    "/version",
    response_model=SystemVersionResponse,
    summary="API and schema versions",
)
async def version_info(request: Request) -> SystemVersionResponse:
    active_model = None
    try:
        reg_svc = request.app.state.model_registry_service
        model = await reg_svc.get_active()
        if model:
            active_model = {
                "model_name":    model.model_name,
                "model_version": model.model_version,
                "git_commit":    model.git_commit,
                "registered_at": str(model.registered_at),
                "is_active":     model.is_active,
            }
    except Exception:
        pass

    return SystemVersionResponse(
        versions=VersionInfo(
            app_version=settings.APP_VERSION,
            app_env=settings.APP_ENV,
            decision_schema_version="decision_fusion_v1",
        ),
        active_model=active_model,
    )


@router.get(
    "/logs",
    response_model=SystemLogsResponse,
    summary="Recent system log entries",
)
async def system_logs(
    request: Request,
    level:     str | None = Query(None, description="Filter by level: INFO, WARNING, ERROR"),
    component: str | None = Query(None, description="Filter by component name"),
    limit:     int        = Query(50, ge=1, le=500),
) -> SystemLogsResponse:
    from src.database.session import async_session_factory
    from src.database.repositories.system_log_repo import SystemLogRepository

    async with async_session_factory() as session:
        repo = SystemLogRepository(session)
        logs = await repo.recent(limit=limit, level=level, component=component)

    entries = [
        {
            "id":             str(e.id),
            "logged_at":      e.logged_at,
            "level":          e.level,
            "component":      e.component,
            "event_type":     e.event_type,
            "message":        e.message,
            "details":        e.details,
            "correlation_id": e.correlation_id,
        }
        for e in logs
    ]

    return SystemLogsResponse(logs=entries, total=len(entries))


@router.post("/debug/trigger-dfe", include_in_schema=False)
async def debug_trigger_dfe(request: Request):
    """Manually trigger DFE and return result or error."""
    import traceback
    svc = request.app.state.prediction_service
    eng = request.app.state.inference_engine

    latest = eng.latest_result()
    if latest is None:
        return {"error": "No inference result available — run inference first"}

    try:
        await svc._trigger_dfe(latest)
        from decision_fusion.recommendation_cache.cache import decision_cache
        d = decision_cache.current
        return {
            "ok": True,
            "recommendation": str(d.recommendation) if d else None,
            "confidence": d.decision_confidence if d else None,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
