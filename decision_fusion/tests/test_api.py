"""HTTP endpoint tests for the DFE API."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from decision_fusion.api.router import dfe_router
from decision_fusion.models.enums import (
    ConsensusLevel,
    MarketBiasEnum,
    Recommendation,
    RecommendationStrength,
)
from decision_fusion.recommendation_cache.cache import DecisionCache
from decision_fusion.schema.decision_object import DecisionObject


def _make_decision(rec: str = "BUY") -> DecisionObject:
    now = datetime.now(timezone.utc)
    return DecisionObject(
        recommendation           = Recommendation(rec),
        recommendation_strength  = RecommendationStrength.STRONG,
        decision_confidence      = 72.0,
        agreement_score          = 80.0,
        conflict_score           = 10.0,
        consensus_level          = ConsensusLevel.STRONG,
        technical_alignment      = 0.72,
        fundamental_alignment    = 0.60,
        market_bias              = MarketBiasEnum.BULLISH if rec == "BUY" else MarketBiasEnum.BEARISH,
        primary_reasons          = ["ML BUY 75%", "EIE BULLISH"],
        supporting_evidence      = ["ML BUY 75%"],
        conflicting_reasons      = [],
        confidence_drivers       = ["ML anchor: BULLISH 75%", "Final confidence: 72.0"],
        risk_factors             = [],
        generated_at             = now,
        expires_at               = now + timedelta(hours=1),
        has_ml                   = True,
        has_eie                  = True,
        has_mia                  = True,
    )


class _FakeDFE:
    def health(self) -> dict:
        return {
            "status": "operational",
            "running": True,
            "schema_version": "decision_fusion_v1",
            "current_recommendation": "BUY",
            "recommendation_strength": "STRONG",
            "recommendation_age_s": 5.0,
            "time_until_expiry_s": 3595.0,
            "is_expired": False,
            "agreement_score": 80.0,
            "conflict_score": 10.0,
            "decision_confidence": 72.0,
            "avg_processing_ms": 4.5,
            "total_decisions": 3,
            "cache_size": 3,
        }


@asynccontextmanager
async def _lifespan_with_decision(app: FastAPI):
    cache = DecisionCache()
    await cache.store(_make_decision("BUY"))
    # Monkey-patch the module-level singleton used by the endpoints
    import decision_fusion.recommendation_cache.cache as cache_mod
    original = cache_mod.decision_cache
    cache_mod.decision_cache = cache
    app.state.dfe = _FakeDFE()
    yield
    cache_mod.decision_cache = original


@asynccontextmanager
async def _lifespan_empty(app: FastAPI):
    yield  # No DFE in app.state, no decisions in cache


@pytest.fixture
def app_with_decision():
    app = FastAPI(lifespan=_lifespan_with_decision)
    app.include_router(dfe_router)
    return app


@pytest.fixture
def app_empty():
    app = FastAPI(lifespan=_lifespan_empty)
    app.include_router(dfe_router)
    return app


@pytest_asyncio.fixture
async def client_with_decision(app_with_decision):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_decision), base_url="http://test"
    ) as c:
        yield c


@pytest_asyncio.fixture
async def client_empty(app_empty):
    async with AsyncClient(
        transport=ASGITransport(app=app_empty), base_url="http://test"
    ) as c:
        yield c


# ── /decision/current ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_current_returns_200_with_decision(client_with_decision):
    resp = await client_with_decision.get("/decision/current")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_current_has_required_fields(client_with_decision):
    resp = await client_with_decision.get("/decision/current")
    data = resp.json()
    assert "decision" in data
    d = data["decision"]
    assert d["recommendation"] == "BUY"
    assert "decision_confidence" in d
    assert "agreement_score" in d
    assert "conflict_score" in d
    assert "primary_reasons" in d
    assert "expires_at" in d


@pytest.mark.asyncio
async def test_current_no_old_field_names(client_with_decision):
    resp = await client_with_decision.get("/decision/current")
    d = resp.json().get("decision", {})
    assert "time_horizon" not in d
    assert "summary" not in d
    assert "generated_at_unix" not in d


@pytest.mark.asyncio
async def test_current_null_when_no_decision(client_empty):
    resp = await client_empty.get("/decision/current")
    assert resp.status_code == 200
    assert resp.json()["decision"] is None


# ── /decision/history ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_returns_200(client_with_decision):
    resp = await client_with_decision.get("/decision/history")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_history_contains_decisions(client_with_decision):
    resp = await client_with_decision.get("/decision/history")
    data = resp.json()
    assert "decisions" in data
    assert data["total"] >= 1


@pytest.mark.asyncio
async def test_history_empty_when_no_decisions(client_empty):
    resp = await client_empty.get("/decision/history")
    data = resp.json()
    assert data["total"] == 0


# ── /decision/confidence ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_confidence_returns_200(client_with_decision):
    resp = await client_with_decision.get("/decision/confidence")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_confidence_has_drivers(client_with_decision):
    resp = await client_with_decision.get("/decision/confidence")
    data = resp.json()
    assert "decision_confidence" in data
    assert "confidence_drivers" in data
    assert data["has_current_decision"] is True


@pytest.mark.asyncio
async def test_confidence_empty_when_no_decision(client_empty):
    resp = await client_empty.get("/decision/confidence")
    data = resp.json()
    assert data["has_current_decision"] is False


# ── /decision/agreement ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agreement_returns_200(client_with_decision):
    resp = await client_with_decision.get("/decision/agreement")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_agreement_has_scores(client_with_decision):
    resp = await client_with_decision.get("/decision/agreement")
    data = resp.json()
    assert "agreement_score" in data
    assert "conflict_score" in data
    assert data["has_current_decision"] is True


# ── /decision/health ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200(client_with_decision):
    resp = await client_with_decision.get("/decision/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_operational(client_with_decision):
    resp = await client_with_decision.get("/decision/health")
    data = resp.json()
    assert data["running"] is True
    assert data["status"] == "operational"
    assert "schema_version" in data
    assert data["schema_version"] == "decision_fusion_v1"


@pytest.mark.asyncio
async def test_health_offline_when_no_dfe(client_empty):
    resp = await client_empty.get("/decision/health")
    data = resp.json()
    assert data["status"] == "offline"
    assert data["running"] is False
