"""
DecisionObject — the official output contract of the Decision Fusion Engine.

This is the final, versioned, explainable recommendation produced by MFIP.
Downstream consumers (frontend, future execution engine) always receive this schema.

Schema version: decision_fusion_v1
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from decision_fusion.models.enums import (
    ConsensusLevel,
    MarketBiasEnum,
    Recommendation,
    RecommendationStrength,
)
from decision_fusion.utils.config import dfe_config


class DecisionObject(BaseModel):
    """
    The canonical Decision Object produced by the Decision Fusion Engine.

    Every field is populated deterministically from evidence — no AI inference
    occurs inside the DFE. This object is the official MFIP output contract.
    """

    # ── Versioning ────────────────────────────────────────────────────────────
    decision_schema_version: str = dfe_config.DECISION_SCHEMA_VERSION

    # ── Identity ──────────────────────────────────────────────────────────────
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ── Core Recommendation ───────────────────────────────────────────────────
    recommendation:          Recommendation
    recommendation_strength: RecommendationStrength

    # ── Confidence ────────────────────────────────────────────────────────────
    decision_confidence: float = Field(ge=0.0, le=100.0)  # 0–100

    # ── Agreement ─────────────────────────────────────────────────────────────
    agreement_score:  float = Field(ge=0.0, le=100.0)  # 0–100
    conflict_score:   float = Field(ge=0.0, le=100.0)  # 0–100
    consensus_level:  ConsensusLevel

    # ── Alignment ─────────────────────────────────────────────────────────────
    # Signed alignment scores: +1.0 = fully bullish, -1.0 = fully bearish, 0 = neutral
    technical_alignment:    float = Field(ge=-1.0, le=1.0)  # From ML
    fundamental_alignment:  float = Field(ge=-1.0, le=1.0)  # From EIE

    # ── Market Bias (from MIA if available, otherwise derived from evidence) ──
    market_bias: MarketBiasEnum

    # ── Explanation (from Explanation Engine) ─────────────────────────────────
    primary_reasons:     List[str]
    supporting_evidence: List[str]
    conflicting_reasons: List[str]
    confidence_drivers:  List[str]
    risk_factors:        List[str]

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    generated_at: datetime
    expires_at:   datetime

    # ── Source Availability (metadata for consumers) ──────────────────────────
    has_ml:  bool = False
    has_eie: bool = False
    has_mia: bool = False

    class Config:
        use_enum_values = True
