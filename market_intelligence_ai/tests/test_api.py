"""Tests for the MIA API endpoints using a test FastAPI application."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from market_intelligence_ai.agent.market_agent import MarketIntelligenceAgent
from market_intelligence_ai.ai_cache.cache import AICache
from market_intelligence_ai.ai_gateway.gateway import AIGateway
from market_intelligence_ai.api.router import mia_router
from market_intelligence_ai.market_context_compiler.compiler import MarketContextCompiler
from market_intelligence_ai.providers.mock_provider import MockProvider
from market_intelligence_ai.utils.config import mia_config


def _make_fake_engine():
    provider = MockProvider(bias="BULLISH")
    gateway  = AIGateway(provider)
    cache    = AICache()
    compiler = MarketContextCompiler()
    agent    = MarketIntelligenceAgent(gateway=gateway, context_compiler=compiler, cache=cache)

    class FakeEngine:
        _agent            = agent
        _context_compiler = compiler
        _analyses         = []

        @property
        def agent(self):
            return self._agent

        async def _build_eie_snapshot(self):
            from market_intelligence_ai.market_context_compiler.context_models import EIESnapshot
            return EIESnapshot()

        async def _store(self, result):
            self._analyses.append(result)

        async def get_recent_analyses(self, limit=20):
            return self._analyses[-limit:]

        def health(self):
            return {
                "running":         True,
                "provider":        "mock",
                "model":           mia_config.MIA_ANALYSIS_MODEL,
                "groq_configured": False,
                "circuit_state":   "CLOSED",
                "cache_stats":     {"hit_rate": 0.0, "total_entries": 0},
                "gateway_metrics": {"total_requests": 0, "failed_requests": 0, "avg_latency_ms": None},
                "analyses_stored": 0,
            }

    return FakeEngine()


@asynccontextmanager
async def _test_lifespan(app: FastAPI):
    app.state.mia = _make_fake_engine()
    yield


@pytest.fixture
def test_app():
    app = FastAPI(lifespan=_test_lifespan)
    app.include_router(mia_router)
    return app


@pytest_asyncio.fixture
async def client(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
        yield c


# ── analyse/event ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_event_returns_200(client):
    response = await client.post("/intelligence/analyse/event", json={
        "event_id": "NFP_TEST",
        "title":    "Non-Farm Payrolls",
        "currency": "USD",
        "actual":   "256K",
        "forecast": "220K",
        "surprise_class": "LARGE",
    })
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert data["analysis"]["market_bias"] == "BULLISH"


@pytest.mark.asyncio
async def test_analyse_event_output_has_required_fields(client):
    response = await client.post("/intelligence/analyse/event", json={
        "event_id": "CPI_TEST",
        "title":    "CPI",
        "currency": "USD",
    })
    data = response.json()["analysis"]
    assert data["analysis_schema_version"] == mia_config.ANALYSIS_SCHEMA_VERSION
    assert "risk_level" in data
    assert "expected_duration" in data
    assert "market_summary" in data
    assert "timestamp" in data
    # Old field names must not be present
    assert "time_horizon" not in data
    assert "summary" not in data
    assert "generated_at" not in data


# ── analyse/headline ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_headline_returns_200(client):
    response = await client.post("/intelligence/analyse/headline", json={
        "headline_id":         "HL_TEST",
        "headline":            "Fed signals no rate cuts this year",
        "source":              "Bloomberg",
        "affected_currencies": ["USD"],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["analysis"]["market_bias"] == "BULLISH"


# ── analyses list ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_analyses_empty(client):
    response = await client.get("/intelligence/analyses")
    assert response.status_code == 200
    data = response.json()
    assert data["analyses"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_analyses_after_event(client):
    await client.post("/intelligence/analyse/event", json={
        "event_id": "GDP_TEST", "title": "GDP", "currency": "USD",
    })
    response = await client.get("/intelligence/analyses")
    data = response.json()
    assert data["total"] >= 1


# ── ai-health ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ai_health_returns_200(client):
    response = await client.get("/intelligence/ai-health")
    assert response.status_code == 200
    data = response.json()
    assert data["running"] is True
    assert "circuit_state" in data
    assert "cache_hit_rate" in data


@pytest.mark.asyncio
async def test_ai_health_no_engine():
    @asynccontextmanager
    async def no_engine_lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=no_engine_lifespan)
    app.include_router(mia_router)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/intelligence/ai-health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "offline"
        assert data["running"] is False
