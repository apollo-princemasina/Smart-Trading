"""Model registry endpoints — version tracking and governance."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.models_registry import (
    ModelListResponse,
    ModelRegistrationRequest,
    ModelRegistryOut,
    ModelResponse,
)

router = APIRouter(prefix="/models", tags=["Model Registry"])


def _to_out(m) -> ModelRegistryOut:
    return ModelRegistryOut(
        id=str(m.id),
        registered_at=m.registered_at,
        model_name=m.model_name,
        model_version=m.model_version,
        bundle_path=m.bundle_path,
        git_commit=m.git_commit,
        feature_schema_version=m.feature_schema_version,
        label_version=m.label_version,
        decision_schema_version=m.decision_schema_version,
        pipeline_version=m.pipeline_version,
        training_start=m.training_start,
        training_end=m.training_end,
        training_dataset=m.training_dataset,
        feature_count=m.feature_count,
        accuracy=m.accuracy,
        precision_buy=m.precision_buy,
        recall_buy=m.recall_buy,
        f1_buy=m.f1_buy,
        precision_sell=m.precision_sell,
        recall_sell=m.recall_sell,
        f1_sell=m.f1_sell,
        is_active=m.is_active,
        notes=m.notes,
        metrics=m.metrics,
    )


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List all registered model versions",
)
async def list_models(request: Request) -> ModelListResponse:
    svc = request.app.state.model_registry_service
    models = await svc.list_all()
    return ModelListResponse(models=[_to_out(m) for m in models], total=len(models))


@router.get(
    "/active",
    response_model=ModelResponse,
    summary="Currently active model with full governance metadata",
)
async def active_model(request: Request) -> ModelResponse:
    svc = request.app.state.model_registry_service
    model = await svc.get_active()
    return ModelResponse(model=_to_out(model) if model else None)


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Get a model version by ID",
)
async def get_model(model_id: str, request: Request) -> ModelResponse:
    svc = request.app.state.model_registry_service
    model = await svc.get_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return ModelResponse(model=_to_out(model))


@router.post(
    "/register",
    response_model=ModelResponse,
    status_code=201,
    summary="Register a new model version",
    description="Deactivates the current active model and registers the new one.",
)
async def register_model(
    body: ModelRegistrationRequest, request: Request
) -> ModelResponse:
    svc = request.app.state.model_registry_service
    model = await svc.register(**body.model_dump())
    return ModelResponse(model=_to_out(model))
