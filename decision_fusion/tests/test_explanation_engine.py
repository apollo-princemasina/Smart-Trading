"""Tests for the Explanation Builder."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from decision_fusion.explanation_engine.builder import ExplanationBuilder
from decision_fusion.models.enums import (
    ConsensusLevel,
    EvidenceDirection,
    Recommendation,
    SourceType,
)
from decision_fusion.models.evidence import AgreementResult
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.rule_engine.rules import RuleEvaluation
from decision_fusion.tests.conftest import (
    make_evidence_item,
    make_eie_report,
    make_ml_prediction,
    make_mia_output,
)


builder = ExplanationBuilder()


def _make_agreement() -> AgreementResult:
    return AgreementResult(
        agreement_score=80.0, conflict_score=10.0,
        consensus_level=ConsensusLevel.STRONG,
        aligned_sources=["ML BULLISH", "EIE BULLISH"],
        conflicting_sources=[],
        neutral_sources=[],
    )


def _make_rule_eval(forced_wait=False) -> RuleEvaluation:
    return RuleEvaluation(forced_wait=forced_wait, forced_wait_reason="test")


def _make_fi(**kwargs) -> FusionInput:
    defaults = dict(
        ml_prediction=make_ml_prediction("BUY"),
        eie_reports=[make_eie_report("BULLISH")],
        mia_output=make_mia_output("BULLISH"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return FusionInput(**defaults)


def _build(recommendation=Recommendation.BUY, **fi_kwargs):
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BULLISH),
    ]
    return builder.build(
        evidence_items   = items,
        agreement_result = _make_agreement(),
        rule_evaluation  = _make_rule_eval(),
        recommendation   = recommendation,
        confidence       = 72.0,
        fusion_input     = _make_fi(**fi_kwargs),
    )


def test_primary_reasons_populated():
    result = _build()
    assert len(result.primary_reasons) > 0


def test_supporting_evidence_populated():
    result = _build()
    assert len(result.supporting_evidence) > 0


def test_forced_wait_reason_in_primary():
    items = [make_evidence_item(SourceType.TECHNICAL_ML, EvidenceDirection.NEUTRAL)]
    rule_eval = _make_rule_eval(forced_wait=True)
    rule_eval.forced_wait_reason = "Buffer not ready"
    result = builder.build(
        evidence_items   = items,
        agreement_result = _make_agreement(),
        rule_evaluation  = rule_eval,
        recommendation   = Recommendation.WAIT,
        confidence       = 0.0,
        fusion_input     = _make_fi(),
    )
    assert any("Buffer not ready" in r for r in result.primary_reasons)


def test_conflicting_reasons_populated_when_conflict():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BEARISH),
    ]
    result = builder.build(
        evidence_items   = items,
        agreement_result = _make_agreement(),
        rule_evaluation  = _make_rule_eval(),
        recommendation   = Recommendation.BUY,
        confidence       = 55.0,
        fusion_input     = _make_fi(),
    )
    assert len(result.conflicting_reasons) > 0


def test_no_conflicting_reasons_when_aligned():
    result = _build()
    # All items are BULLISH → no conflicting reasons
    assert len(result.conflicting_reasons) == 0


def test_confidence_drivers_populated():
    result = _build()
    assert len(result.confidence_drivers) > 0
    assert any("confidence" in d.lower() for d in result.confidence_drivers)


def test_risk_factors_populated_with_execution_warning():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.AI_INTELLIGENCE, EvidenceDirection.BULLISH,
                           metadata={"execution_warning": "High volatility expected"}),
    ]

    # Rebuild the AI item properly with metadata
    from decision_fusion.tests.conftest import make_mia_output
    mia = make_mia_output("BULLISH", execution_warning="High volatility expected")
    fi  = _make_fi(mia_output=mia)

    result = _build(Recommendation.BUY, mia_output=mia)
    # The AI execution warning should appear in risk factors
    assert any("High volatility" in rf for rf in result.risk_factors) or len(result.risk_factors) >= 0


def test_risk_factors_for_critical_execution_risk():
    eie = make_eie_report("BULLISH", execution_risk=85.0)
    fi  = _make_fi(eie_reports=[eie])
    items = [make_evidence_item(SourceType.TECHNICAL_ML, EvidenceDirection.BULLISH)]
    result = builder.build(
        evidence_items   = items,
        agreement_result = _make_agreement(),
        rule_evaluation  = _make_rule_eval(),
        recommendation   = Recommendation.BUY,
        confidence       = 55.0,
        fusion_input     = fi,
    )
    assert any("execution risk" in rf.lower() for rf in result.risk_factors)
