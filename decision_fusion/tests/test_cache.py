"""Tests for the Decision Cache."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from decision_fusion.recommendation_cache.cache import DecisionCache
from decision_fusion.models.enums import (
    ConsensusLevel,
    MarketBiasEnum,
    Recommendation,
    RecommendationStrength,
)
from decision_fusion.schema.decision_object import DecisionObject


def _make_decision(
    rec: str = "BUY",
    strength: str = "STRONG",
    confidence: float = 72.0,
    expires_offset_s: float = 3600.0,
) -> DecisionObject:
    now = datetime.now(timezone.utc)
    return DecisionObject(
        recommendation           = Recommendation(rec),
        recommendation_strength  = RecommendationStrength(strength),
        decision_confidence      = confidence,
        agreement_score          = 80.0,
        conflict_score           = 10.0,
        consensus_level          = ConsensusLevel.STRONG,
        technical_alignment      = 0.75,
        fundamental_alignment    = 0.60,
        market_bias              = MarketBiasEnum.BULLISH,
        primary_reasons          = ["ML BUY"],
        supporting_evidence      = ["ML BUY 75%"],
        conflicting_reasons      = [],
        confidence_drivers       = ["Base: ML 75%"],
        risk_factors             = [],
        generated_at             = now,
        expires_at               = now + timedelta(seconds=expires_offset_s),
    )


@pytest.mark.asyncio
async def test_store_and_retrieve_current():
    cache = DecisionCache()
    d = _make_decision()
    await cache.store(d)
    assert cache.current is not None
    assert cache.current.recommendation == Recommendation.BUY


@pytest.mark.asyncio
async def test_previous_promoted_on_second_store():
    cache = DecisionCache()
    d1 = _make_decision("BUY")
    d2 = _make_decision("SELL")
    await cache.store(d1)
    await cache.store(d2)
    assert cache.previous.recommendation == Recommendation.BUY
    assert cache.current.recommendation == Recommendation.SELL


@pytest.mark.asyncio
async def test_history_grows():
    cache = DecisionCache()
    for _ in range(5):
        await cache.store(_make_decision())
    assert cache.size() == 5


@pytest.mark.asyncio
async def test_history_limit():
    from decision_fusion.utils.config import dfe_config
    cache = DecisionCache()
    # Fill beyond maxlen
    for _ in range(dfe_config.DFE_HISTORY_MAX_SIZE + 10):
        await cache.store(_make_decision())
    assert cache.size() == dfe_config.DFE_HISTORY_MAX_SIZE


@pytest.mark.asyncio
async def test_get_history_newest_first():
    cache = DecisionCache()
    await cache.store(_make_decision("BUY"))
    await cache.store(_make_decision("SELL"))
    history = cache.get_history(2)
    assert history[0].recommendation == Recommendation.SELL


@pytest.mark.asyncio
async def test_is_expired_for_fresh_decision():
    cache = DecisionCache()
    await cache.store(_make_decision(expires_offset_s=3600.0))
    assert cache.is_expired() is False


@pytest.mark.asyncio
async def test_is_expired_for_old_decision():
    cache = DecisionCache()
    await cache.store(_make_decision(expires_offset_s=-1.0))  # already expired
    assert cache.is_expired() is True


@pytest.mark.asyncio
async def test_is_expired_when_no_current():
    cache = DecisionCache()
    assert cache.is_expired() is True


@pytest.mark.asyncio
async def test_invalidate_clears_current():
    cache = DecisionCache()
    await cache.store(_make_decision())
    await cache.invalidate()
    assert cache.current is None


@pytest.mark.asyncio
async def test_age_seconds_increases():
    cache = DecisionCache()
    await cache.store(_make_decision())
    age = cache.age_seconds()
    assert age is not None
    assert age >= 0.0


@pytest.mark.asyncio
async def test_seconds_until_expiry_positive():
    cache = DecisionCache()
    await cache.store(_make_decision(expires_offset_s=3600.0))
    remaining = cache.seconds_until_expiry()
    assert remaining is not None
    assert remaining > 0.0


@pytest.mark.asyncio
async def test_stats_populated():
    cache = DecisionCache()
    await cache.store(_make_decision())
    stats = cache.stats()
    assert stats["has_current"] is True
    assert stats["history_size"] == 1
    assert stats["is_expired"] is False
