"""End-to-end tests for the Decision Fusion Engine."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from decision_fusion.engine import DecisionFusionEngine
from decision_fusion.models.enums import Recommendation, RecommendationStrength
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.recommendation_cache.cache import DecisionCache
from decision_fusion.tests.conftest import (
    make_eie_report,
    make_ml_prediction,
    make_mia_output,
    full_fusion_input,
    bearish_fusion_input,
    no_evidence_fusion_input,
    ml_hold_fusion_input,
    conflict_fusion_input,
)


def _fresh_engine() -> DecisionFusionEngine:
    """Create an engine with a fresh private cache (avoids test cross-contamination)."""
    engine = DecisionFusionEngine()
    # Override the module-level decision_cache with a fresh instance
    from decision_fusion import recommendation_cache
    import decision_fusion.engine as engine_module
    fresh_cache = DecisionCache()
    engine_module.decision_cache = fresh_cache
    engine._fresh_cache = fresh_cache  # track for assertions
    return engine


@pytest.mark.asyncio
async def test_process_returns_decision_object():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY", 0.75),
        eie_reports=[make_eie_report("BULLISH")],
        mia_output=make_mia_output("BULLISH"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)
    assert decision is not None
    assert decision.recommendation in (Recommendation.BUY, Recommendation.SELL, Recommendation.WAIT)


@pytest.mark.asyncio
async def test_triple_bullish_produces_buy():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY", 0.75),
        eie_reports=[make_eie_report("BULLISH")],
        mia_output=make_mia_output("BULLISH"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)
    assert decision.recommendation == Recommendation.BUY


@pytest.mark.asyncio
async def test_triple_bearish_produces_sell():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("SELL", 0.70),
        eie_reports=[make_eie_report("BEARISH")],
        mia_output=make_mia_output("BEARISH"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)
    assert decision.recommendation == Recommendation.SELL


@pytest.mark.asyncio
async def test_no_evidence_produces_wait():
    fi = FusionInput(
        ml_prediction=None,
        eie_reports=[],
        mia_output=None,
        buffer_ready=False,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)
    assert decision.recommendation == Recommendation.WAIT


@pytest.mark.asyncio
async def test_ml_hold_produces_wait():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("HOLD", 0.40),
        eie_reports=[make_eie_report("BULLISH")],
        mia_output=make_mia_output("BULLISH"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)
    assert decision.recommendation == Recommendation.WAIT


@pytest.mark.asyncio
async def test_decision_has_all_required_fields():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)

    assert decision.decision_schema_version is not None
    assert decision.decision_id is not None
    assert decision.recommendation is not None
    assert decision.recommendation_strength is not None
    assert decision.decision_confidence is not None
    assert decision.agreement_score is not None
    assert decision.conflict_score is not None
    assert decision.consensus_level is not None
    assert decision.market_bias is not None
    assert decision.generated_at is not None
    assert decision.expires_at is not None
    assert isinstance(decision.primary_reasons, list)
    assert isinstance(decision.risk_factors, list)


@pytest.mark.asyncio
async def test_decision_confidence_in_range():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY", 0.75),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)
    assert 0.0 <= decision.decision_confidence <= 100.0


@pytest.mark.asyncio
async def test_expires_at_in_future():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY", 0.75),
        eie_reports=[make_eie_report("BULLISH")],
        mia_output=make_mia_output("BULLISH"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)
    assert decision.expires_at > decision.generated_at


@pytest.mark.asyncio
async def test_schema_version_correct():
    fi = FusionInput(buffer_ready=False, current_time=datetime.now(timezone.utc))
    engine = DecisionFusionEngine()
    decision = await engine.process(fi)
    assert decision.decision_schema_version == "decision_fusion_v1"


@pytest.mark.asyncio
async def test_health_returns_operational_after_process():
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY"),
        buffer_ready=True,
        current_time=datetime.now(timezone.utc),
    )
    engine = DecisionFusionEngine()
    engine._running = True
    await engine.process(fi)
    health = engine.health()
    assert health["running"] is True
    assert health["total_decisions"] >= 1


@pytest.mark.asyncio
async def test_process_deterministic():
    """Same input must always produce the same recommendation."""
    fi = FusionInput(
        ml_prediction=make_ml_prediction("BUY", 0.72),
        eie_reports=[make_eie_report("BULLISH", 0.80, 70.0, 60.0)],
        mia_output=make_mia_output("BULLISH", 0.70),
        buffer_ready=True,
        current_time=datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    engine1 = DecisionFusionEngine()
    engine2 = DecisionFusionEngine()
    d1 = await engine1.process(fi)
    d2 = await engine2.process(fi)
    assert d1.recommendation == d2.recommendation
    assert d1.recommendation_strength == d2.recommendation_strength
    assert abs(d1.decision_confidence - d2.decision_confidence) < 0.01
