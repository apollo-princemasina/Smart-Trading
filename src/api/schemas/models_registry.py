"""Pydantic schemas for model registry endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelRegistryOut(BaseModel):
    id:                      str
    registered_at:           Any
    model_name:              str
    model_version:           str
    bundle_path:             str
    git_commit:              str | None
    feature_schema_version:  str | None
    label_version:           str | None
    decision_schema_version: str | None
    pipeline_version:        str | None
    training_start:          str | None
    training_end:            str | None
    training_dataset:        str | None
    feature_count:           int
    accuracy:                float | None
    precision_buy:           float | None
    recall_buy:              float | None
    f1_buy:                  float | None
    precision_sell:          float | None
    recall_sell:             float | None
    f1_sell:                 float | None
    is_active:               bool
    notes:                   str | None
    metrics:                 dict[str, Any] | None


class ModelRegistrationRequest(BaseModel):
    model_name:              str
    model_version:           str
    bundle_path:             str
    git_commit:              str | None = None
    feature_schema_version:  str | None = None
    label_version:           str | None = None
    decision_schema_version: str | None = None
    pipeline_version:        str | None = None
    training_start:          str | None = None
    training_end:            str | None = None
    training_dataset:        str | None = None
    feature_count:           int = 247
    accuracy:                float | None = None
    precision_buy:           float | None = None
    recall_buy:              float | None = None
    f1_buy:                  float | None = None
    precision_sell:          float | None = None
    recall_sell:             float | None = None
    f1_sell:                 float | None = None
    notes:                   str | None = None
    metrics:                 dict[str, Any] | None = None


class ModelListResponse(BaseModel):
    models: list[ModelRegistryOut]
    total:  int


class ModelResponse(BaseModel):
    model: ModelRegistryOut | None
