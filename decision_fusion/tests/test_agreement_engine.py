"""Tests for the Agreement Calculator."""
from __future__ import annotations

import pytest

from decision_fusion.agreement_engine.calculator import AgreementCalculator
from decision_fusion.models.enums import ConsensusLevel, EvidenceDirection, SourceType
from decision_fusion.tests.conftest import make_evidence_item


calculator = AgreementCalculator()


def test_all_bullish_high_agreement():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.AI_INTELLIGENCE, EvidenceDirection.BULLISH),
    ]
    result = calculator.compute(items)
    assert result.agreement_score == pytest.approx(100.0, abs=1.0)
    assert result.conflict_score == pytest.approx(0.0, abs=1.0)


def test_all_bearish_high_agreement():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BEARISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BEARISH),
    ]
    result = calculator.compute(items)
    assert result.agreement_score == pytest.approx(100.0, abs=1.0)
    assert result.conflict_score == pytest.approx(0.0, abs=1.0)


def test_bullish_vs_bearish_high_conflict():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BEARISH),
    ]
    result = calculator.compute(items)
    assert result.conflict_score > 50.0
    assert result.agreement_score < result.conflict_score


def test_one_source_returns_no_agreement():
    items = [make_evidence_item(EvidenceDirection.BULLISH)]
    result = calculator.compute(items)
    assert result.agreement_score == 0.0
    assert result.conflict_score == 0.0


def test_absent_sources_excluded():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.ABSENT),
        make_evidence_item(SourceType.AI_INTELLIGENCE, EvidenceDirection.BULLISH),
    ]
    result = calculator.compute(items)
    # Only 2 directional sources → should agree
    assert result.agreement_score > 50.0


def test_uncertain_sources_excluded_from_pairs():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.UNCERTAIN),
        make_evidence_item(SourceType.AI_INTELLIGENCE, EvidenceDirection.BULLISH),
    ]
    result = calculator.compute(items)
    assert result.agreement_score > 50.0


def test_consensus_very_strong_on_full_agreement():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.AI_INTELLIGENCE, EvidenceDirection.BULLISH),
    ]
    result = calculator.compute(items)
    assert result.consensus_level == ConsensusLevel.VERY_STRONG


def test_consensus_weak_on_full_conflict():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BEARISH),
    ]
    result = calculator.compute(items)
    assert result.consensus_level in (ConsensusLevel.WEAK, ConsensusLevel.MODERATE)


def test_neutral_paired_with_directional_no_conflict():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.NEUTRAL),
    ]
    result = calculator.compute(items)
    # Neutral ↔ Bullish is not a conflict
    assert result.conflict_score == pytest.approx(0.0, abs=1.0)


def test_aligned_sources_populated():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BULLISH),
    ]
    result = calculator.compute(items)
    assert len(result.aligned_sources) >= 1


def test_conflicting_sources_populated():
    items = [
        make_evidence_item(SourceType.TECHNICAL_ML,    EvidenceDirection.BULLISH),
        make_evidence_item(SourceType.FUNDAMENTAL_EIE, EvidenceDirection.BEARISH),
    ]
    result = calculator.compute(items)
    assert len(result.conflicting_sources) >= 1
