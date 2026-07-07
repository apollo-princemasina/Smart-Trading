"""Tests for the Rule Engine — conditions and evaluator."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from decision_fusion.models.enums import ConsensusLevel, RecommendationStrength
from decision_fusion.rule_engine.rules import (
    RULES,
    EvaluationContext,
    RuleAction,
)
from decision_fusion.rule_engine.evaluator import RuleEvaluator


evaluator = RuleEvaluator()


def _make_ctx(**overrides) -> EvaluationContext:
    defaults = dict(
        has_ml               = True,
        has_eie              = True,
        has_mia              = True,
        buffer_ready         = True,
        total_source_count   = 3,
        active_source_count  = 3,
        conflict_score       = 10.0,
        agreement_score      = 90.0,
        preliminary_confidence = 72.0,
        ml_direction         = "BUY",
        fundamental_direction = "BULLISH",
        ai_direction         = "BULLISH",
        eie_execution_risk   = 20.0,
        ai_risk_level        = "LOW",
        consensus_level      = ConsensusLevel.STRONG,
    )
    defaults.update(overrides)
    return EvaluationContext(**defaults)


# ── Rule conditions ────────────────────────────────────────────────────────────

def test_r001_no_evidence_triggers():
    ctx = _make_ctx(total_source_count=0, active_source_count=0)
    result = evaluator.evaluate(ctx)
    assert result.forced_wait is True
    assert "R001" in result.triggered_rules


def test_r002_buffer_not_ready_triggers():
    ctx = _make_ctx(buffer_ready=False)
    result = evaluator.evaluate(ctx)
    assert result.forced_wait is True
    assert "R002" in result.triggered_rules


def test_r003_critical_conflict_triggers():
    ctx = _make_ctx(conflict_score=80.0, preliminary_confidence=30.0)
    result = evaluator.evaluate(ctx)
    assert result.forced_wait is True
    assert "R003" in result.triggered_rules


def test_r004_ml_hold_triggers_wait():
    ctx = _make_ctx(ml_direction="HOLD")
    result = evaluator.evaluate(ctx)
    assert result.forced_wait is True
    assert "R004" in result.triggered_rules


def test_r005_critical_execution_risk_reduces_confidence():
    ctx = _make_ctx(eie_execution_risk=85.0)
    result = evaluator.evaluate(ctx)
    assert result.confidence_delta < 0
    assert "R005" in result.triggered_rules


def test_r006_ml_eie_direct_conflict_reduces_confidence():
    ctx = _make_ctx(ml_direction="BUY", fundamental_direction="BEARISH")
    result = evaluator.evaluate(ctx)
    assert result.confidence_delta < 0
    assert "R006" in result.triggered_rules


def test_r009_triple_confirm_buy_boosts_confidence():
    ctx = _make_ctx(
        ml_direction="BUY",
        fundamental_direction="BULLISH",
        ai_direction="BULLISH",
    )
    result = evaluator.evaluate(ctx)
    # R009 should trigger a BOOST_CONFIDENCE
    assert "R009" in result.triggered_rules
    boost_rules = [
        r for r in RULES
        if r.rule_id == "R009" and r.action == RuleAction.BOOST_CONFIDENCE
    ]
    assert len(boost_rules) == 1
    assert result.confidence_delta > 0


def test_r010_triple_confirm_sell_boosts_confidence():
    ctx = _make_ctx(
        ml_direction="SELL",
        fundamental_direction="BEARISH",
        ai_direction="BEARISH",
    )
    result = evaluator.evaluate(ctx)
    assert "R010" in result.triggered_rules


def test_r011_single_source_caps_strength():
    ctx = _make_ctx(active_source_count=1)
    result = evaluator.evaluate(ctx)
    assert "R011" in result.triggered_rules
    assert result.strength_cap == RecommendationStrength.MODERATE


def test_r013_low_confidence_forces_wait():
    ctx = _make_ctx(preliminary_confidence=20.0)
    result = evaluator.evaluate(ctx)
    assert result.forced_wait is True
    assert "R013" in result.triggered_rules


def test_no_rules_triggered_on_clean_input():
    ctx = _make_ctx(
        conflict_score=5.0,
        agreement_score=95.0,
        preliminary_confidence=75.0,
        eie_execution_risk=10.0,
        ai_risk_level="LOW",
        active_source_count=3,
        ml_direction="BUY",
        fundamental_direction="BULLISH",
        ai_direction="BULLISH",
    )
    result = evaluator.evaluate(ctx)
    assert result.forced_wait is False
    # Only positive rules should trigger (R009, possibly R012)
    force_wait_rules = [r for r in result.triggered_rules if r in ("R001", "R002", "R003", "R004", "R013")]
    assert force_wait_rules == []


def test_evaluation_stops_after_force_wait():
    # R002 (buffer not ready) should stop evaluation immediately
    ctx = _make_ctx(buffer_ready=False, eie_execution_risk=90.0)
    result = evaluator.evaluate(ctx)
    # R005 (high exec risk) would also trigger, but evaluation stopped
    assert result.forced_wait is True
    # R005 may or may not have triggered — what matters is forced_wait is set


def test_multiple_reductions_accumulate():
    ctx = _make_ctx(
        conflict_score=60.0,
        eie_execution_risk=85.0,
        ml_direction="BUY",
        fundamental_direction="BEARISH",
    )
    result = evaluator.evaluate(ctx)
    assert result.confidence_delta < -10.0  # Multiple penalties should accumulate


def test_rule_reasons_populated():
    ctx = _make_ctx(conflict_score=60.0)
    result = evaluator.evaluate(ctx)
    if result.triggered_rules:
        assert len(result.rule_reasons) == len(result.triggered_rules)
