"""Pydantic schemas for the dashboard endpoint."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DecisionSummary(BaseModel):
    recommendation:    str
    strength:          str
    confidence:        float
    agreement_score:   float
    conflict_score:    float
    consensus_level:   str
    market_bias:       str
    primary_reasons:   list[str]
    risk_factors:      list[str]
    generated_at:      str | None
    expires_at:        str | None
    is_expired:        bool
    age_seconds:       float | None
    schema_version:    str


class PredictionSummary(BaseModel):
    id:          str
    direction:   str
    confidence:  float
    regime:      str | None
    close:       float | None
    tp_price:    float | None
    sl_price:    float | None
    atr_pips:    float | None
    session:     str | None
    signal_time: str | None


class RegimeSummary(BaseModel):
    regime:  str | None
    scores:  dict[str, Any] = {}
    bias:    str | None
    session: str | None


class MIASummary(BaseModel):
    market_bias:       str | None
    confidence:        float | None
    risk_level:        str | None
    market_summary:    str | None
    expected_duration: str | None
    is_fallback:       bool


class EIEEventItem(BaseModel):
    title:    str
    currency: str
    impact:   str
    time:     str
    forecast: str | None = None
    previous: str | None = None


class EIESummary(BaseModel):
    active_count:      int
    has_active_events: bool
    execution_risk:    float
    upcoming:          list[EIEEventItem] = []


class BufferSummary(BaseModel):
    ready:          bool = False
    timeframes:     dict[str, Any] = {}


class SystemSummary(BaseModel):
    uptime_seconds:        float
    websocket_connections: int
    scheduler_running:     bool


class DashboardResponse(BaseModel):
    decision:           DecisionSummary | None
    latest_prediction:  PredictionSummary | None
    market_regime:      RegimeSummary | None
    mia_summary:        MIASummary | None
    eie_summary:        EIESummary | None
    buffer_status:      dict[str, Any] | None
    system_summary:     SystemSummary
