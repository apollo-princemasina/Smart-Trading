"""DFE API router — mounts all decision endpoints under /decision."""
from fastapi import APIRouter

from decision_fusion.api.endpoints.agreement  import router as agreement_router
from decision_fusion.api.endpoints.confidence import router as confidence_router
from decision_fusion.api.endpoints.decision   import router as decision_router
from decision_fusion.api.endpoints.health     import router as health_router

dfe_router = APIRouter(prefix="/decision", tags=["Decision Fusion Engine"])

dfe_router.include_router(decision_router)
dfe_router.include_router(confidence_router)
dfe_router.include_router(agreement_router)
dfe_router.include_router(health_router)
