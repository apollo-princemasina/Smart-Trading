"""DFE API request/response Pydantic models."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from decision_fusion.models.enums import (
    ConsensusLevel,
    MarketBiasEnum,
    Recommendation,
    RecommendationStrength,
)
from decision_fusion.utils.config import dfe_config


class DecisionOut(BaseModel):
    """Single DecisionObject serialised for API consumers."""
    decision_schema_version:  str
    decision_id:              str
    recommendation:           Recommendation
    recommendation_strength:  RecommendationStrength
    decision_confidence:      float
    agreement_score:          float
    conflict_score:           float
    consensus_level:          ConsensusLevel
    technical_alignment:      float
    fundamental_alignment:    float
    market_bias:              MarketBiasEnum
    primary_reasons:          List[str]
    supporting_evidence:      List[str]
    conflicting_reasons:      List[str]
    confidence_drivers:       List[str]
    risk_factors:             List[str]
    generated_at:             datetime
    expires_at:               datetime
    has_ml:                   bool
    has_eie:                  bool
    has_mia:                  bool

    model_config = {"from_attributes": True}


class DecisionResponse(BaseModel):
    """Current decision envelope."""
    decision:           Optional[DecisionOut]
    is_expired:         bool
    age_seconds:        Optional[float]
    seconds_until_expiry: Optional[float]


class DecisionHistoryResponse(BaseModel):
    """List of recent decisions."""
    decisions: List[DecisionOut]
    total:     int


class ConfidenceBreakdownResponse(BaseModel):
    """Detailed breakdown of how the current confidence score was constructed."""
    decision_confidence:  Optional[float]
    ml_confidence:        Optional[float]    # 0–100 from ML model
    eie_confidence:       Optional[float]    # aggregated EIE direction confidence 0–100
    ai_confidence:        Optional[float]    # MIA model confidence 0–100
    agreement_score:      Optional[float]
    conflict_score:       Optional[float]
    consensus_level:      Optional[ConsensusLevel]
    confidence_drivers:   List[str]
    has_current_decision: bool


class AgreementBreakdownResponse(BaseModel):
    """Detailed breakdown of inter-source agreement."""
    agreement_score:     Optional[float]
    conflict_score:      Optional[float]
    consensus_level:     Optional[ConsensusLevel]
    aligned_sources:     List[str]
    conflicting_sources: List[str]
    neutral_sources:     List[str]
    has_current_decision: bool


class DFEHealthResponse(BaseModel):
    """DFE subsystem health dashboard."""
    status:                   str     # "operational" | "degraded" | "offline"
    running:                  bool
    schema_version:           str
    current_recommendation:   Optional[str]
    recommendation_strength:  Optional[str]
    recommendation_age_s:     Optional[float]
    time_until_expiry_s:      Optional[float]
    is_expired:               bool
    agreement_score:          Optional[float]
    conflict_score:           Optional[float]
    decision_confidence:      Optional[float]
    avg_processing_ms:        Optional[float]
    total_decisions:          int
    cache_size:               int
