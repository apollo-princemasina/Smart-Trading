"""
Integration tests for the EIE API layer.

Uses a stripped FastAPI test app that runs only the EIE endpoints.
The FF connector cache is mocked with static MFIPEvent data.
No ML, database, or Twelve Data calls required.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator
from unittest.mock import patch, AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from economic_intelligence.api.router import eie_router
from economic_intelligence.engine import EconomicIntelligenceEngine
from economic_intelligence.intelligence_cache.cache import IntelligenceCache


# ── Minimal test events ───────────────────────────────────────────────────────

from market_intel.models.enums import ImpactLevel, EventStatus, EventCategory, Provider
from market_intel.models.event import MFIPEvent
from forex_factory_connector.cache.memory_cache import WeekCache

now = datetime.now(timezone.utc)

_MOCK_EVENTS = [
    MFIPEvent(
        event_id="test_nfp",
        provider=Provider.FOREX_FACTORY,
        provider_event_id="prov_nfp",
        title="US Non-Farm Employment Change",
        currency="USD", country="US",
        timestamp_utc=now - timedelta(hours=1),
        is_all_day=False,
        impact=ImpactLevel.HIGH,
        is_high_impact=True, is_speech=False,
        category=EventCategory.EMPLOYMENT,
        forecast="185K", previous="177K", actual="206K",
        status=EventStatus.RELEASED,
        last_updated=now,
    ),
    MFIPEvent(
        event_id="test_cpi",
        provider=Provider.FOREX_FACTORY,
        provider_event_id="prov_cpi",
        title="German CPI m/m",
        currency="EUR", country="EU",
        timestamp_utc=now + timedelta(hours=3),
        is_all_day=False,
        impact=ImpactLevel.HIGH,
        is_high_impact=True, is_speech=False,
        category=EventCategory.INFLATION,
        forecast="0.2%", previous="0.0%", actual=None,
        status=EventStatus.SCHEDULED,
        last_updated=now,
    ),
]

_MOCK_WEEK_CACHE = WeekCache(events=_MOCK_EVENTS, fetched_at=now)


@asynccontextmanager
async def _test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    engine = EconomicIntelligenceEngine()
    await engine.startup()
    app.state.eie_engine = engine
    yield
    await engine.shutdown()


@pytest.fixture
def test_app():
    app = FastAPI(lifespan=_test_lifespan)
    app.include_router(eie_router, prefix="/api/v1")
    return app


@pytest_asyncio.fixture
async def client(test_app):
    with patch(
        "economic_intelligence.engine.EconomicIntelligenceEngine._fetch_all_events",
        new=AsyncMock(return_value=_MOCK_EVENTS),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as ac:
            yield ac


# ── /context ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_endpoint_returns_200(client):
    resp = await client.get("/api/v1/intelligence/context")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_context_schema(client):
    resp = await client.get("/api/v1/intelligence/context")
    body = resp.json()
    required = {"context", "active_events", "upcoming_events", "total_active", "total_upcoming", "generated_at"}
    assert required.issubset(body.keys()), f"Missing keys: {required - body.keys()}"
    ctx = body["context"]
    ctx_required = {
        "execution_risk", "execution_readiness", "risk_rationale",
        "readiness_rationale", "is_market_open", "is_holiday",
    }
    assert ctx_required.issubset(ctx.keys())


@pytest.mark.asyncio
async def test_context_scores_in_range(client):
    resp = await client.get("/api/v1/intelligence/context")
    ctx = resp.json()["context"]
    assert 0 <= ctx["execution_risk"]      <= 100
    assert 0 <= ctx["execution_readiness"] <= 100


# ── /execution-risk ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execution_risk_endpoint(client):
    resp = await client.get("/api/v1/intelligence/execution-risk")
    assert resp.status_code == 200
    body = resp.json()
    assert "execution_risk"  in body
    assert "rationale"       in body
    assert "is_market_open"  in body
    assert 0 <= body["execution_risk"] <= 100


# ── /readiness ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_readiness_endpoint(client):
    resp = await client.get("/api/v1/intelligence/readiness")
    assert resp.status_code == 200
    body = resp.json()
    assert "execution_readiness" in body
    assert 0 <= body["execution_readiness"] <= 100


# ── /active-events ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_active_events_endpoint(client):
    resp = await client.get("/api/v1/intelligence/active-events")
    assert resp.status_code == 200
    body = resp.json()
    assert "events" in body
    assert "count"  in body
    # Released NFP event should appear (it has influence)
    if body["count"] > 0:
        event = body["events"][0]
        assert event["is_released"] is True


@pytest.mark.asyncio
async def test_active_events_currency_filter(client):
    resp = await client.get("/api/v1/intelligence/active-events?currency=USD")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert all(e["currency"] == "USD" for e in events)


# ── /upcoming-events ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upcoming_events_endpoint(client):
    resp = await client.get("/api/v1/intelligence/upcoming-events")
    assert resp.status_code == 200
    body = resp.json()
    assert "events"      in body
    assert "hours_ahead" in body
    # Scheduled German CPI should appear
    if body["count"] > 0:
        event = body["events"][0]
        assert event["is_released"] is False


@pytest.mark.asyncio
async def test_upcoming_events_hours_filter(client):
    resp = await client.get("/api/v1/intelligence/upcoming-events?hours_ahead=2.0")
    body = resp.json()
    assert body["hours_ahead"] == 2.0
    # German CPI is 3h away — should NOT appear in 2h window
    events = body["events"]
    assert not any(e["event_title"] == "German CPI m/m" for e in events)


# ── /economic-summary ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_economic_summary_endpoint(client):
    resp = await client.get("/api/v1/intelligence/economic-summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "currencies"    in body
    assert "total_reports" in body
    assert body["total_reports"] > 0


@pytest.mark.asyncio
async def test_economic_summary_currency_entries(client):
    resp = await client.get("/api/v1/intelligence/economic-summary")
    currencies = {c["currency"] for c in resp.json()["currencies"]}
    # Both USD (NFP) and EUR (German CPI) should appear
    assert "USD" in currencies
    assert "EUR" in currencies


@pytest.mark.asyncio
async def test_economic_summary_direction_field(client):
    resp = await client.get("/api/v1/intelligence/economic-summary")
    for entry in resp.json()["currencies"]:
        assert entry["dominant_direction"] in ("BULLISH", "BEARISH", "NEUTRAL", "UNCERTAIN")
        assert 0 <= entry["avg_confidence"] <= 1
        assert 0 <= entry["avg_impact_score"] <= 100


# ── EIE output schema completeness ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_intelligence_report_schema_completeness(client):
    """Every report must contain all canonical EIE output fields."""
    resp = await client.get("/api/v1/intelligence/context")
    body = resp.json()
    all_events = body["active_events"] + body["upcoming_events"]

    if not all_events:
        pytest.skip("No events in cache — cannot verify schema")

    required_fields = {
        "report_id", "event_id", "generated_at",
        "event_title", "currency", "country", "timestamp_utc",
        "is_released", "importance", "event_type",
        "impact_score",
        "surprise", "pct_surprise", "surprise_class", "surprise_direction",
        "economic_direction", "direction_confidence", "direction_rationale",
        "remaining_influence", "event_age_hours",
        "time_to_event",
        "execution_risk", "execution_readiness",
        "confidence", "last_updated",
    }

    for event in all_events:
        missing = required_fields - set(event.keys())
        assert not missing, f"Report missing fields: {missing}"
