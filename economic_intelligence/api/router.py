"""
EIE API router — extends /api/v1/intelligence with Phase 3 endpoints.

Route map (all under /intelligence)
------------------------------------
  GET /intelligence/context          Full economic context snapshot
  GET /intelligence/execution-risk   Current execution risk score
  GET /intelligence/readiness        Current execution readiness score
  GET /intelligence/active-events    Released events with remaining influence
  GET /intelligence/upcoming-events  Scheduled events (next N hours)
  GET /intelligence/economic-summary Per-currency economic direction summary
"""
from fastapi import APIRouter

from economic_intelligence.api.endpoints import context, risk, events, summary

eie_router = APIRouter(prefix="/intelligence", tags=["Economic Intelligence"])

eie_router.include_router(context.router)
eie_router.include_router(risk.router)
eie_router.include_router(events.router)
eie_router.include_router(summary.router)
