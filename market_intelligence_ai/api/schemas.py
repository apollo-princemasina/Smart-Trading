"""
MIA API response schemas — Pydantic models for all HTTP responses.

These wrap MarketIntelligenceOutput with API-level envelope fields.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from market_intelligence_ai.schema.market_intelligence import MarketIntelligenceOutput
from market_intelligence_ai.models.enums import MarketBias, Importance, TimeHorizon


class AnalyseEventRequest(BaseModel):
    """Request body for POST /intelligence/analyse/event."""
    event_id:          str
    title:             str
    currency:          str
    importance:        str = "MEDIUM"
    forecast:          Optional[str] = None
    actual:            Optional[str] = None
    previous:          Optional[str] = None
    surprise_class:    str = "NONE"
    surprise_direction: str = "IN_LINE"
    economic_direction: str = "UNCERTAIN"


class AnalyseHeadlineRequest(BaseModel):
    """Request body for POST /intelligence/analyse/headline."""
    headline_id:          str
    headline:             str
    source:               str
    affected_currencies:  List[str]


class AnalysisResponse(BaseModel):
    """Single analysis response envelope."""
    analysis:     MarketIntelligenceOutput
    request_id:   str
    timestamp:    datetime


class AnalysisListResponse(BaseModel):
    """List of recent analyses."""
    analyses: List[MarketIntelligenceOutput]
    total:    int


class AIHealthResponse(BaseModel):
    """AI subsystem health status."""
    status:           str   # "ok" | "degraded" | "offline"
    running:          bool
    provider:         str
    model:            str
    groq_configured:  bool
    circuit_state:    str   # "CLOSED" | "OPEN" | "HALF_OPEN"
    cache_hit_rate:   float
    total_requests:   int
    failed_requests:  int
    avg_latency_ms:   Optional[float]
    analyses_stored:  int
    cache_total_entries: int
