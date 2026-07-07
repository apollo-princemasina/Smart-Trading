"""GET /intelligence/ai-health — AI subsystem health dashboard."""
from __future__ import annotations

from fastapi import APIRouter, Request

from market_intelligence_ai.api.schemas import AIHealthResponse
from market_intelligence_ai.utils.config import mia_config

router = APIRouter()


@router.get("/ai-health", response_model=AIHealthResponse, summary="AI subsystem health")
async def ai_health(request: Request):
    """
    Reports the health and operational status of the Market Intelligence AI Layer.

    Includes provider status, circuit breaker state, cache hit rate, and
    gateway metrics.
    """
    engine = getattr(request.app.state, "mia", None)

    if engine is None:
        return AIHealthResponse(
            status          = "offline",
            running         = False,
            provider        = "groq",
            model           = mia_config.MIA_ANALYSIS_MODEL,
            groq_configured = mia_config.groq_configured,
            circuit_state   = "UNKNOWN",
            cache_hit_rate  = 0.0,
            total_requests  = 0,
            failed_requests = 0,
            avg_latency_ms  = None,
            analyses_stored = 0,
            cache_total_entries = 0,
        )

    h       = engine.health()
    metrics = h.get("gateway_metrics", {})
    cache   = h.get("cache_stats", {})

    status = "ok"
    if not h.get("groq_configured"):
        status = "degraded"
    elif h.get("circuit_state") == "OPEN":
        status = "degraded"

    return AIHealthResponse(
        status           = status,
        running          = h.get("running", False),
        provider         = h.get("provider", "groq"),
        model            = h.get("model", mia_config.MIA_ANALYSIS_MODEL),
        groq_configured  = h.get("groq_configured", False),
        circuit_state    = h.get("circuit_state", "UNKNOWN"),
        cache_hit_rate   = cache.get("hit_rate", 0.0),
        total_requests   = metrics.get("total_requests", 0),
        failed_requests  = metrics.get("failed_requests", 0),
        avg_latency_ms   = metrics.get("avg_latency_ms"),
        analyses_stored  = h.get("analyses_stored", 0),
        cache_total_entries = cache.get("total_entries", 0),
    )
