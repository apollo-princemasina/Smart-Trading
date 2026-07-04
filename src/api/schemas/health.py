from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class BufferStatus(BaseModel):
    timeframe:  str
    size:       int
    expected:   int
    oldest_bar: Optional[datetime] = None
    newest_bar: Optional[datetime] = None
    ready:      bool


class ModelStatus(BaseModel):
    loaded:        bool
    bundle_path:   str
    feature_count: int
    model_name:    Optional[str] = None
    loaded_at:     Optional[datetime] = None


class DBStatus(BaseModel):
    connected:       bool
    prediction_count: int


class HealthResponse(BaseModel):
    status:       str          # "healthy" | "degraded" | "unhealthy"
    version:      str
    environment:  str
    timestamp:    datetime
    uptime_s:     float
    buffer:       list[BufferStatus]
    model:        ModelStatus
    database:     DBStatus
    scheduler_running: bool
