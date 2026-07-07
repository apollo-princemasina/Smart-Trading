"""
MarketIntelligenceOutput — the single structured output contract for all analysis.

This is the only output model produced by the Market Intelligence Agent.
All downstream systems consume this schema regardless of what triggered the analysis
(economic event, headline, or combined context).

Schema: market_intelligence_ai_v1
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from market_intelligence_ai.models.enums import MarketBias, Importance, TimeHorizon, RiskLevel
from market_intelligence_ai.utils.config import mia_config


class MarketIntelligenceOutput(BaseModel):
    """
    Structured market intelligence produced by the autonomous Market Intelligence Agent.

    Never contains trading recommendations. The five internal reasoning perspectives
    (Economist, FX Strategist, Microstructure Analyst, Risk Manager, Communicator)
    are synthesised here — only the final structured conclusion is exposed.
    """

    # ── Versioning ────────────────────────────────────────────────────────────
    analysis_schema_version: str = mia_config.ANALYSIS_SCHEMA_VERSION
    context_schema_version:  str = mia_config.CONTEXT_SCHEMA_VERSION
    provider:                str = ""   # e.g. "groq:llama-3.3-70b-versatile"

    # ── Core intelligence fields (match spec exactly) ─────────────────────────
    market_bias:               MarketBias
    affected_currencies:       List[str]
    importance:                Importance
    confidence:                float = Field(ge=0.0, le=1.0)
    expected_duration:         TimeHorizon              # IMMEDIATE | SHORT_TERM | MEDIUM_TERM | LONG_TERM
    supports_existing_bias:    bool
    contradicts_existing_bias: bool
    risk_level:                RiskLevel                # Risk Manager perspective output
    execution_warning:         Optional[str] = None
    market_summary:            str                      # Communicator perspective output

    # ── Audit fields (not included in AI output — injected server-side) ─────────
    timestamp:   Optional[datetime] = None   # stamped by agent after validation
    latency_ms:  float = 0.0                 # stamped by gateway after validation
    is_fallback: bool = False                # True when AI failed and fallback was returned
    cache_hit:   bool = False                # True when served from AICache
