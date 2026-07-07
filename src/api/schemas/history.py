"""Pydantic schemas for unified history endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PredictionHistoryItem(BaseModel):
    id:            str
    signal_time:   Any
    symbol:        str
    timeframe:     str | None
    # Adjusted (session-weighted) output
    direction:     str
    confidence:    float
    # Raw model output before session weighting / demotion
    raw_direction: str | None
    raw_confidence: float | None
    demoted:       bool
    # Full probability vector
    prob_buy:      float | None
    prob_sell:     float | None
    prob_hold:     float | None
    # Session
    session:       str | None
    session_mult:  float | None
    # Market context
    regime:        str | None
    close:         float | None
    tp_price:      float | None
    sl_price:      float | None
    tp_pips:       float | None
    sl_pips:       float | None
    atr_pips:      float | None


class DecisionHistoryItem(BaseModel):
    id:              str
    decision_id:     str
    generated_at:    Any
    expires_at:      Any
    recommendation:  str
    strength:        str
    confidence:      float
    agreement_score: float
    conflict_score:  float
    consensus_level: str
    market_bias:     str
    primary_reasons: list[str]
    risk_factors:    list[str]
    has_ml:          bool
    has_eie:         bool
    has_mia:         bool
    schema_version:  str


class PredictionHistoryResponse(BaseModel):
    predictions: list[PredictionHistoryItem]
    total:       int
    page:        int
    page_size:   int


class DecisionHistoryResponse(BaseModel):
    decisions:  list[DecisionHistoryItem]
    total:      int
    page:       int
    page_size:  int


class CombinedHistoryItem(BaseModel):
    record_type:   str   # prediction | decision
    timestamp:     Any
    id:            str
    # Polymorphic fields — present depending on record_type
    direction:      str | None = None
    recommendation: str | None = None
    confidence:     float | None = None
    strength:       str | None = None
    agreement_score: float | None = None
    regime:         str | None = None
    symbol:         str | None = None


class CombinedHistoryResponse(BaseModel):
    items:    list[CombinedHistoryItem]
    page:     int
    page_size: int
