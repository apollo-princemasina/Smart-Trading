"""Pydantic schemas for system endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    status:  str           # ok | degraded | error | stopped
    details: dict[str, Any] = {}


class SystemHealthResponse(BaseModel):
    status:          str    # operational | degraded
    uptime_seconds:  float
    components:      dict[str, dict[str, Any]]


class SystemStatusResponse(BaseModel):
    status:                str
    uptime_seconds:        float
    engines_online:        list[str]
    engine_count:          int


class VersionInfo(BaseModel):
    app_version:             str
    app_env:                 str
    decision_schema_version: str
    api_version:             str = "v1"


class SystemVersionResponse(BaseModel):
    versions: VersionInfo
    active_model: dict[str, Any] | None


class SystemLogEntry(BaseModel):
    id:             str
    logged_at:      Any
    level:          str
    component:      str
    event_type:     str
    message:        str
    details:        dict[str, Any] | None
    correlation_id: str | None


class SystemLogsResponse(BaseModel):
    logs:  list[SystemLogEntry]
    total: int
