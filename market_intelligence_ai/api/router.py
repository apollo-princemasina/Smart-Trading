"""MIA API router — mounts all intelligence endpoints under /intelligence."""
from fastapi import APIRouter

from market_intelligence_ai.api.endpoints.analysis import router as analysis_router
from market_intelligence_ai.api.endpoints.health import router as health_router

mia_router = APIRouter(prefix="/intelligence", tags=["Market Intelligence AI"])

mia_router.include_router(analysis_router)
mia_router.include_router(health_router)
