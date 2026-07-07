"""Shared fixtures for DFE tests."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from decision_fusion.models.enums import EvidenceDirection, SourceType
from decision_fusion.models.evidence import EvidenceItem
from decision_fusion.models.fusion_input import FusionInput


# ── ML prediction fixtures ────────────────────────────────────────────────────

def make_ml_prediction(
    direction: str = "BUY",
    confidence: float = 0.75,
    raw_confidence: float = 0.78,
    regime: str = "EXPANSION",
    session: str = "LONDON_OPEN",
) -> dict:
    return {
        "direction":      direction,
        "confidence":     confidence,
        "raw_confidence": raw_confidence,
        "prob_buy":       confidence if direction == "BUY" else 0.1,
        "prob_sell":      confidence if direction == "SELL" else 0.1,
        "prob_hold":      1 - confidence - 0.1 if direction != "HOLD" else confidence,
        "regime":         regime,
        "regime_scores":  {"EXPANSION": 0.6, "CONSOLIDATION": 0.3, "MANIPULATION": 0.1},
        "session":        session,
        "session_mult":   1.0,
        "model_version":  "lgbm_v2",
        "signal_time":    datetime.now(timezone.utc),
        "symbol":         "EURUSD",
        "timeframe":      "M15",
        "close":          1.08500,
        "atr_pips":       8.5,
        "tp_price":       1.08755,
        "sl_price":       1.08373,
        "tp_pips":        25.5,
        "sl_pips":        12.7,
    }


# ── EIE report mock ────────────────────────────────────────────────────────────

def make_eie_report(
    direction: str = "BULLISH",
    direction_confidence: float = 0.80,
    impact_score: float = 75.0,
    remaining_influence: float = 65.0,
    importance: str = "HIGH",
    execution_risk: float = 20.0,
    execution_readiness: float = 80.0,
    event_title: str = "Non-Farm Payrolls",
    currency: str = "USD",
) -> Any:
    r = MagicMock()
    r.economic_direction        = MagicMock()
    r.economic_direction.value  = direction
    r.direction_confidence      = direction_confidence
    r.impact_score              = impact_score
    r.remaining_influence       = remaining_influence
    r.importance                = MagicMock()
    r.importance.value          = importance
    r.execution_risk            = execution_risk
    r.execution_readiness       = execution_readiness
    r.event_title               = event_title
    r.currency                  = currency
    r.generated_at              = datetime.now(timezone.utc)
    r.last_updated              = datetime.now(timezone.utc)
    r.surprise_class            = MagicMock()
    r.surprise_class.value      = "LARGE"
    r.direction_rationale       = "Strong beat on payrolls"
    return r


# ── MIA output mock ────────────────────────────────────────────────────────────

def make_mia_output(
    market_bias: str = "BULLISH",
    confidence: float = 0.72,
    importance: str = "HIGH",
    risk_level: str = "LOW",
    execution_warning: str = None,
    market_summary: str = "USD strengthens on strong NFP beat.",
    is_fallback: bool = False,
) -> Any:
    m = MagicMock()
    m.market_bias               = MagicMock()
    m.market_bias.value         = market_bias
    m.confidence                = confidence
    m.importance                = MagicMock()
    m.importance.value          = importance
    m.risk_level                = MagicMock()
    m.risk_level.value          = risk_level
    m.execution_warning         = execution_warning
    m.market_summary            = market_summary
    m.is_fallback               = is_fallback
    m.supports_existing_bias    = True
    m.contradicts_existing_bias = False
    m.affected_currencies       = ["USD"]
    m.timestamp                 = datetime.now(timezone.utc)
    return m


# ── FusionInput helpers ───────────────────────────────────────────────────────

@pytest.fixture
def full_fusion_input() -> FusionInput:
    """All three sources available, all bullish — should produce BUY."""
    return FusionInput(
        ml_prediction = make_ml_prediction("BUY", 0.75),
        eie_reports   = [make_eie_report("BULLISH")],
        mia_output    = make_mia_output("BULLISH"),
        latest_close  = 1.08500,
        buffer_ready  = True,
        current_time  = datetime.now(timezone.utc),
    )


@pytest.fixture
def bearish_fusion_input() -> FusionInput:
    """All three sources bearish — should produce SELL."""
    return FusionInput(
        ml_prediction = make_ml_prediction("SELL", 0.70),
        eie_reports   = [make_eie_report("BEARISH")],
        mia_output    = make_mia_output("BEARISH"),
        latest_close  = 1.08500,
        buffer_ready  = True,
        current_time  = datetime.now(timezone.utc),
    )


@pytest.fixture
def no_evidence_fusion_input() -> FusionInput:
    """No sources available — should produce WAIT."""
    return FusionInput(
        ml_prediction = None,
        eie_reports   = [],
        mia_output    = None,
        latest_close  = None,
        buffer_ready  = False,
        current_time  = datetime.now(timezone.utc),
    )


@pytest.fixture
def ml_hold_fusion_input() -> FusionInput:
    """ML returns HOLD — should produce WAIT."""
    return FusionInput(
        ml_prediction = make_ml_prediction("HOLD", 0.40),
        eie_reports   = [make_eie_report("BULLISH")],
        mia_output    = make_mia_output("BULLISH"),
        latest_close  = 1.08500,
        buffer_ready  = True,
        current_time  = datetime.now(timezone.utc),
    )


@pytest.fixture
def conflict_fusion_input() -> FusionInput:
    """ML BUY but EIE and MIA both BEARISH — conflict."""
    return FusionInput(
        ml_prediction = make_ml_prediction("BUY", 0.62),
        eie_reports   = [make_eie_report("BEARISH")],
        mia_output    = make_mia_output("BEARISH"),
        latest_close  = 1.08500,
        buffer_ready  = True,
        current_time  = datetime.now(timezone.utc),
    )


# ── Evidence item helpers ──────────────────────────────────────────────────────

def make_evidence_item(
    source: SourceType = SourceType.TECHNICAL_ML,
    direction: EvidenceDirection = EvidenceDirection.BULLISH,
    confidence: float = 0.75,
    reliability: float = 0.85,
    importance: float = 1.0,
) -> EvidenceItem:
    return EvidenceItem(
        source      = source,
        direction   = direction,
        confidence  = confidence,
        reliability = reliability,
        importance  = importance,
        timestamp   = datetime.now(timezone.utc),
        label       = f"{source.value} {direction.value}",
    )
