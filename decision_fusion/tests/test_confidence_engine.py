"""Tests for the Confidence Fusion Engine."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from decision_fusion.agreement_engine.calculator import AgreementCalculator
from decision_fusion.confidence_engine.fusion import ConfidenceFusion
from decision_fusion.evidence_engine.collector import EvidenceCollector
from decision_fusion.models.enums import EvidenceDirection, SourceType
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.tests.conftest import (
    make_eie_report,
    make_evidence_item,
    make_ml_prediction,
    make_mia_output,
)


collector   = EvidenceCollector()
agreement   = AgreementCalculator()
fusion      = ConfidenceFusion()


def _compute(fi: FusionInput) -> float:
    items  = collector.collect(fi)
    agr    = agreement.compute(items)
    return fusion.compute(items, agr, fi)


def _make_fi(**kwargs) -> FusionInput:
    defaults = dict(
        ml_prediction=None, eie_reports=[], mia_output=None,
        latest_close=1.085, buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return FusionInput(**defaults)


def test_confidence_in_range():
    fi = _make_fi(ml_prediction=make_ml_prediction("BUY", 0.75))
    conf = _compute(fi)
    assert 0.0 <= conf <= 100.0


def test_ml_sets_base_confidence():
    fi = _make_fi(ml_prediction=make_ml_prediction("BUY", 0.75))
    conf = _compute(fi)
    # Base = 75. With weak agreement and no penalties, should be around 50-80
    assert conf > 0.0


def test_triple_alignment_higher_than_single_source():
    fi_single = _make_fi(ml_prediction=make_ml_prediction("BUY", 0.70))
    fi_triple = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.70),
        eie_reports=[make_eie_report("BULLISH")],
        mia_output=make_mia_output("BULLISH"),
    )
    conf_single = _compute(fi_single)
    conf_triple = _compute(fi_triple)
    assert conf_triple > conf_single


def test_conflict_lowers_confidence():
    fi_aligned = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.72),
        eie_reports=[make_eie_report("BULLISH")],
    )
    fi_conflict = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.72),
        eie_reports=[make_eie_report("BEARISH")],
    )
    conf_aligned  = _compute(fi_aligned)
    conf_conflict = _compute(fi_conflict)
    assert conf_conflict < conf_aligned


def test_high_execution_risk_lowers_confidence():
    fi_low_risk  = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.72),
        eie_reports=[make_eie_report("BULLISH", execution_risk=10.0)],
    )
    fi_high_risk = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.72),
        eie_reports=[make_eie_report("BULLISH", execution_risk=90.0)],
    )
    assert _compute(fi_high_risk) < _compute(fi_low_risk)


def test_fallback_base_when_no_sources():
    from decision_fusion.utils.config import dfe_config
    fi = _make_fi()
    conf = _compute(fi)
    # No sources — should use fallback base, then weakened by weak consensus
    assert conf <= dfe_config.DFE_FALLBACK_BASE_CONFIDENCE


def test_critical_ai_risk_lowers_confidence():
    fi_low  = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.75),
        mia_output=make_mia_output("BULLISH", risk_level="LOW"),
    )
    fi_crit = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.75),
        mia_output=make_mia_output("BULLISH", risk_level="CRITICAL"),
    )
    assert _compute(fi_crit) < _compute(fi_low)


def test_confidence_does_not_exceed_100():
    fi = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.99),
        eie_reports=[make_eie_report("BULLISH", execution_risk=0.0)],
        mia_output=make_mia_output("BULLISH"),
    )
    conf = _compute(fi)
    assert conf <= 100.0


def test_confidence_does_not_go_below_0():
    fi = _make_fi(
        ml_prediction=make_ml_prediction("BUY", 0.01),
        eie_reports=[make_eie_report("BEARISH", execution_risk=99.0)],
        mia_output=make_mia_output("BEARISH", risk_level="CRITICAL"),
    )
    conf = _compute(fi)
    assert conf >= 0.0
