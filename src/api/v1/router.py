"""API v1 router — mounts all endpoint sub-routers."""
from fastapi import APIRouter

from src.api.v1.endpoints.health       import router as health_router
from src.api.v1.endpoints.predictions  import router as predictions_router
from src.api.v1.endpoints.market       import router as market_router
from forex_factory_connector.api.router import intelligence_router
from economic_intelligence.api.router     import eie_router
from market_intelligence_ai.api.router    import mia_router
from decision_fusion.api.router           import dfe_router

# Application Backend routers (Phase 6)
from src.api.v1.endpoints.dashboard       import router as dashboard_router
from src.api.v1.endpoints.system          import router as system_router
from src.api.v1.endpoints.settings_ep     import router as settings_router
from src.api.v1.endpoints.models_registry import router as models_router
from src.api.v1.endpoints.history         import router as history_router
from src.api.v1.endpoints.auth            import router as auth_router

v1_router = APIRouter(prefix="/api/v1")

# Phase 1 — core inference
v1_router.include_router(health_router)
v1_router.include_router(predictions_router)
v1_router.include_router(market_router)

# Phase 2–5 — intelligence engines
v1_router.include_router(intelligence_router)
v1_router.include_router(eie_router)
v1_router.include_router(mia_router)
v1_router.include_router(dfe_router)

# Phase 6 — application backend
v1_router.include_router(dashboard_router)
v1_router.include_router(system_router)
v1_router.include_router(settings_router)
v1_router.include_router(models_router)
v1_router.include_router(history_router)
v1_router.include_router(auth_router)
