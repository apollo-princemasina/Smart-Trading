"""Tests for the Evidence Collector."""
from __future__ import annotations

import pytest

from decision_fusion.evidence_engine.collector import EvidenceCollector
from decision_fusion.models.enums import EvidenceDirection, SourceType
from decision_fusion.tests.conftest import (
    make_eie_report,
    make_ml_prediction,
    make_mia_output,
)
from decision_fusion.models.fusion_input import FusionInput
from datetime import datetime, timezone


collector = EvidenceCollector()


def _make_fi(**kwargs) -> FusionInput:
    defaults = dict(
        ml_prediction=None, eie_reports=[], mia_output=None,
        latest_close=1.085, buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return FusionInput(**defaults)


# ── ML Evidence ───────────────────────────────────────────────────────────────

def test_ml_absent_when_no_prediction():
    items = collector.collect(_make_fi())
    ml = next(i for i in items if i.source == SourceType.TECHNICAL_ML)
    assert ml.direction == EvidenceDirection.ABSENT


def test_ml_buy_maps_to_bullish():
    fi = _make_fi(ml_prediction=make_ml_prediction("BUY", 0.75))
    items = collector.collect(fi)
    ml = next(i for i in items if i.source == SourceType.TECHNICAL_ML)
    assert ml.direction == EvidenceDirection.BULLISH
    assert ml.confidence == pytest.approx(0.75)


def test_ml_sell_maps_to_bearish():
    fi = _make_fi(ml_prediction=make_ml_prediction("SELL", 0.68))
    items = collector.collect(fi)
    ml = next(i for i in items if i.source == SourceType.TECHNICAL_ML)
    assert ml.direction == EvidenceDirection.BEARISH


def test_ml_hold_maps_to_neutral():
    fi = _make_fi(ml_prediction=make_ml_prediction("HOLD", 0.45))
    items = collector.collect(fi)
    ml = next(i for i in items if i.source == SourceType.TECHNICAL_ML)
    assert ml.direction == EvidenceDirection.NEUTRAL


def test_ml_reliability_set_correctly():
    fi = _make_fi(ml_prediction=make_ml_prediction("BUY", 0.75))
    items = collector.collect(fi)
    ml = next(i for i in items if i.source == SourceType.TECHNICAL_ML)
    assert ml.reliability == pytest.approx(0.85, abs=0.01)


# ── EIE Evidence ──────────────────────────────────────────────────────────────

def test_eie_absent_when_no_reports():
    items = collector.collect(_make_fi())
    eie = next(i for i in items if i.source == SourceType.FUNDAMENTAL_EIE)
    assert eie.direction == EvidenceDirection.ABSENT


def test_eie_bullish_direction():
    fi = _make_fi(eie_reports=[make_eie_report("BULLISH")])
    items = collector.collect(fi)
    eie = next(i for i in items if i.source == SourceType.FUNDAMENTAL_EIE)
    assert eie.direction == EvidenceDirection.BULLISH


def test_eie_bearish_direction():
    fi = _make_fi(eie_reports=[make_eie_report("BEARISH")])
    items = collector.collect(fi)
    eie = next(i for i in items if i.source == SourceType.FUNDAMENTAL_EIE)
    assert eie.direction == EvidenceDirection.BEARISH


def test_eie_low_remaining_influence_produces_absent():
    low_influence = make_eie_report("BULLISH", remaining_influence=5.0)
    fi = _make_fi(eie_reports=[low_influence])
    items = collector.collect(fi)
    eie = next(i for i in items if i.source == SourceType.FUNDAMENTAL_EIE)
    assert eie.direction == EvidenceDirection.ABSENT


def test_eie_multiple_reports_aggregate():
    reports = [
        make_eie_report("BULLISH", impact_score=80, remaining_influence=70),
        make_eie_report("BULLISH", impact_score=60, remaining_influence=50),
    ]
    fi = _make_fi(eie_reports=reports)
    items = collector.collect(fi)
    eie = next(i for i in items if i.source == SourceType.FUNDAMENTAL_EIE)
    assert eie.direction == EvidenceDirection.BULLISH


# ── MIA Evidence ──────────────────────────────────────────────────────────────

def test_mia_absent_when_no_output():
    items = collector.collect(_make_fi())
    mia = next(i for i in items if i.source == SourceType.AI_INTELLIGENCE)
    assert mia.direction == EvidenceDirection.ABSENT


def test_mia_bullish_direction():
    fi = _make_fi(mia_output=make_mia_output("BULLISH"))
    items = collector.collect(fi)
    mia = next(i for i in items if i.source == SourceType.AI_INTELLIGENCE)
    assert mia.direction == EvidenceDirection.BULLISH


def test_mia_fallback_produces_absent():
    fi = _make_fi(mia_output=make_mia_output("BULLISH", is_fallback=True))
    items = collector.collect(fi)
    mia = next(i for i in items if i.source == SourceType.AI_INTELLIGENCE)
    assert mia.direction == EvidenceDirection.ABSENT


def test_mia_risk_level_in_metadata():
    fi = _make_fi(mia_output=make_mia_output("BULLISH", risk_level="HIGH"))
    items = collector.collect(fi)
    mia = next(i for i in items if i.source == SourceType.AI_INTELLIGENCE)
    assert mia.metadata.get("risk_level") == "HIGH"


# ── Full collection ───────────────────────────────────────────────────────────

def test_full_collection_returns_three_items():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY"),
        eie_reports=[make_eie_report("BULLISH")],
        mia_output=make_mia_output("BULLISH"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    items = collector.collect(fi)
    sources = {i.source for i in items}
    assert SourceType.TECHNICAL_ML in sources
    assert SourceType.FUNDAMENTAL_EIE in sources
    assert SourceType.AI_INTELLIGENCE in sources
    assert len(items) == 3


def test_weight_property_positive():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY", 0.80),
        eie_reports=[],
        mia_output=None,
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    items = collector.collect(fi)
    ml = next(i for i in items if i.source == SourceType.TECHNICAL_ML)
    assert ml.weight > 0


def test_directional_weight_bullish_positive():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY", 0.75),
        eie_reports=[],
        mia_output=None,
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    items = collector.collect(fi)
    ml = next(i for i in items if i.source == SourceType.TECHNICAL_ML)
    assert ml.directional_weight > 0


def test_directional_weight_bearish_negative():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("SELL", 0.70),
        eie_reports=[],
        mia_output=None,
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    items = collector.collect(fi)
    ml = next(i for i in items if i.source == SourceType.TECHNICAL_ML)
    assert ml.directional_weight < 0
