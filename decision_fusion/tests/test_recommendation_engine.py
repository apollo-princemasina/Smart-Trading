"""Tests for the Recommendation Generator."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from decision_fusion.models.enums import (
    ConsensusLevel,
    EvidenceDirection,
    Recommendation,
    RecommendationStrength,
    SourceType,
)
from decision_fusion.models.evidence import AgreementResult
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.recommendation_engine.generator import RecommendationGenerator
from decision_fusion.rule_engine.rules import RuleEvaluation
from decision_fusion.tests.conftest import (
    make_evidence_item,
    make_eie_report,
    make_ml_prediction,
)


gen = RecommendationGenerator()


def _make_agreement(
    agreement_score=80.0,
    conflict_score=5.0,
    consensus=ConsensusLevel.STRONG,
) -> AgreementResult:
    return AgreementResult(
        agreement_score     = agreement_score,
        conflict_score      = conflict_score,
        consensus_level     = consensus,
        aligned_sources     = [],
        conflicting_sources = [],
        neutral_sources     = [],
    )


def _make_rule_eval(forced_wait=False, delta=0.0, cap=None) -> RuleEvaluation:
    return RuleEvaluation(
        forced_wait       = forced_wait,
        forced_wait_reason = "test",
        confidence_delta  = delta,
        strength_cap      = cap,
    )


def _bullish_items():
    return [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH, 0.75),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BULLISH, 0.80),
    ]


def _bearish_items():
    return [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BEARISH, 0.70),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BEARISH, 0.72),
    ]


# ── Recommendation direction ───────────────────────────────────────────────────

def test_bullish_evidence_produces_buy():
    result = gen.generate(_bullish_items(), _make_agreement(), 72.0, _make_rule_eval())
    assert result.recommendation == Recommendation.BUY


def test_bearish_evidence_produces_sell():
    result = gen.generate(_bearish_items(), _make_agreement(), 68.0, _make_rule_eval())
    assert result.recommendation == Recommendation.SELL


def test_forced_wait_produces_wait():
    result = gen.generate(_bullish_items(), _make_agreement(), 72.0, _make_rule_eval(forced_wait=True))
    assert result.recommendation == Recommendation.WAIT
    assert result.forced_wait is True


def test_forced_wait_strength_is_weak():
    result = gen.generate(_bullish_items(), _make_agreement(), 72.0, _make_rule_eval(forced_wait=True))
    assert result.strength == RecommendationStrength.WEAK


# ── Strength ──────────────────────────────────────────────────────────────────

def test_high_confidence_produces_strong_recommendation():
    result = gen.generate(
        _bullish_items(), _make_agreement(80, 5, ConsensusLevel.VERY_STRONG), 80.0, _make_rule_eval()
    )
    assert result.strength in (RecommendationStrength.STRONG, RecommendationStrength.VERY_STRONG)


def test_low_confidence_produces_weak_recommendation():
    result = gen.generate(
        _bullish_items(), _make_agreement(40, 20, ConsensusLevel.WEAK), 35.0, _make_rule_eval()
    )
    assert result.strength == RecommendationStrength.WEAK


def test_strength_cap_is_applied():
    result = gen.generate(
        _bullish_items(), _make_agreement(80, 5, ConsensusLevel.STRONG), 80.0,
        _make_rule_eval(cap=RecommendationStrength.MODERATE)
    )
    assert result.strength == RecommendationStrength.MODERATE


def test_confidence_delta_is_applied():
    base = 70.0
    delta = -15.0
    result = gen.generate(_bullish_items(), _make_agreement(), base, _make_rule_eval(delta=delta))
    assert result.confidence == pytest.approx(base + delta, abs=0.1)


# ── Technical alignment ───────────────────────────────────────────────────────

def test_technical_alignment_buy_positive():
    fi = FusionInput(ml_prediction=make_ml_prediction("BUY", 0.75), buffer_ready=True,
                     current_time=datetime.now(timezone.utc))
    assert gen.compute_technical_alignment(fi) > 0.0


def test_technical_alignment_sell_negative():
    fi = FusionInput(ml_prediction=make_ml_prediction("SELL", 0.70), buffer_ready=True,
                     current_time=datetime.now(timezone.utc))
    assert gen.compute_technical_alignment(fi) < 0.0


def test_technical_alignment_hold_zero():
    fi = FusionInput(ml_prediction=make_ml_prediction("HOLD", 0.45), buffer_ready=True,
                     current_time=datetime.now(timezone.utc))
    assert gen.compute_technical_alignment(fi) == pytest.approx(0.0)


def test_fundamental_alignment_bullish_positive():
    fi = FusionInput(eie_reports=[make_eie_report("BULLISH")], buffer_ready=True,
                     current_time=datetime.now(timezone.utc))
    assert gen.compute_fundamental_alignment(fi) > 0.0


def test_fundamental_alignment_bearish_negative():
    fi = FusionInput(eie_reports=[make_eie_report("BEARISH")], buffer_ready=True,
                     current_time=datetime.now(timezone.utc))
    assert gen.compute_fundamental_alignment(fi) < 0.0


# ── Expiry ────────────────────────────────────────────────────────────────────

def test_wait_expires_sooner_than_strong_buy():
    from decision_fusion.utils.config import dfe_config
    now = datetime.now(timezone.utc)
    wait_expiry  = gen.compute_expiry(Recommendation.WAIT, RecommendationStrength.WEAK,  now)
    strong_expiry = gen.compute_expiry(Recommendation.BUY, RecommendationStrength.STRONG, now)
    assert wait_expiry < strong_expiry
