"""API v1 router — mounts all endpoint sub-routers."""
from fastapi import APIRouter

from src.api.v1.endpoints.health      import router as health_router
from src.api.v1.endpoints.predictions import router as predictions_router
from src.api.v1.endpoints.market      import router as market_router

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(health_router)
v1_router.include_router(predictions_router)
v1_router.include_router(market_router)
