"""
Integration tests for the Forex Factory Connector.

Uses a stripped FastAPI test app that runs only the Intelligence Layer —
no database, no ML models, no Twelve Data calls required.

The CDN fetcher is patched with a realistic mock JSON payload so tests are
hermetic and repeatable.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from forex_factory_connector.connector           import ForexFactoryConnector
from forex_factory_connector.api.router          import intelligence_router
from forex_factory_connector.fetcher.cdn_fetcher import CDNFetchResult

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_EVENTS: list[dict] = [
    {
        "title":    "US Non-Farm Employment Change",
        "country":  "USD",
        "date":     (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S"),
        "impact":   "High",
        "forecast": "185K",
        "previous": "177K",
    },
    {
        "title":    "German CPI m/m",
        "country":  "EUR",
        "date":     (datetime.now(timezone.utc) + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S"),
        "impact":   "Medium",
        "forecast": "0.2%",
        "previous": "0.0%",
    },
    {
        "title":    "Fed Chair Powell Speaks",
        "country":  "USD",
        "date":     (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%S"),
        "impact":   "High",
        "forecast": "",
        "previous": "",
    },
]

MOCK_BODY = json.dumps(MOCK_EVENTS).encode()

_MOCK_RESULT = CDNFetchResult(body=MOCK_BODY, etag='"test-etag"', not_modified=False)
_MOCK_304    = CDNFetchResult(body=None,      etag='"test-etag"', not_modified=True)


def _mock_fetch(not_modified: bool = False):
    result = _MOCK_304 if not_modified else _MOCK_RESULT
    return AsyncMock(return_value=result)


@asynccontextmanager
async def _test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    connector = ForexFactoryConnector()
    await connector.startup()
    app.state.ff_connector = connector
    yield
    await connector.shutdown()


@pytest.fixture
def test_app():
    app = FastAPI(lifespan=_test_lifespan)
    app.include_router(intelligence_router, prefix="/api/v1")
    return app


@pytest_asyncio.fixture
async def client(test_app):
    with patch(
        "forex_factory_connector.fetcher.cdn_fetcher.fetch_calendar",
        side_effect=_mock_fetch(),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as ac:
            yield ac


# ── Startup / shutdown lifecycle ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_connector_startup_and_shutdown():
    """Connector starts up cleanly, populates cache, starts scheduler, then shuts down."""
    with patch(
        "forex_factory_connector.fetcher.cdn_fetcher.fetch_calendar",
        side_effect=_mock_fetch(),
    ):
        connector = ForexFactoryConnector()
        assert not connector.is_running

        await connector.startup()
        assert connector.is_running
        assert connector.started_at is not None
        assert connector.uptime_s is not None and connector.uptime_s >= 0

        await connector.shutdown()
        assert not connector.is_running


@pytest.mark.asyncio
async def test_cache_warm_up():
    """Cache is populated for all three weeks during startup."""
    from forex_factory_connector.cache.memory_cache import connector_cache

    with patch(
        "forex_factory_connector.fetcher.cdn_fetcher.fetch_calendar",
        side_effect=_mock_fetch(),
    ):
        connector = ForexFactoryConnector()
        await connector.startup()

        assert connector_cache.is_populated("thisweek")
        assert connector_cache.is_populated("nextweek")
        assert connector_cache.is_populated("lastweek")

        thisweek = await connector_cache.get_calendar("thisweek")
        assert len(thisweek.events) == len(MOCK_EVENTS)

        await connector.shutdown()


@pytest.mark.asyncio
async def test_cache_warm_up_survives_fetch_failure():
    """Connector startup does not raise even if all CDN fetches fail."""
    with patch(
        "forex_factory_connector.fetcher.cdn_fetcher.fetch_calendar",
        side_effect=AsyncMock(side_effect=Exception("CDN unreachable")),
    ):
        connector = ForexFactoryConnector()
        await connector.startup()   # must NOT raise
        await connector.shutdown()


# ── Calendar endpoint ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_calendar_endpoint_returns_events(client):
    resp = await client.get("/api/v1/intelligence/calendar")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == len(MOCK_EVENTS)
    assert body["week"] == "thisweek"
    assert isinstance(body["events"], list)


@pytest.mark.asyncio
async def test_calendar_currency_filter(client):
    resp = await client.get("/api/v1/intelligence/calendar?currency=USD")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert all(e["currency"] == "USD" for e in events)


@pytest.mark.asyncio
async def test_calendar_high_impact_endpoint(client):
    resp = await client.get("/api/v1/intelligence/calendar/high-impact")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert all(e["is_high_impact"] for e in events)


# ── Events endpoints ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_events_today_endpoint(client):
    resp = await client.get("/api/v1/intelligence/events/today")
    assert resp.status_code == 200
    body = resp.json()
    assert "events" in body
    assert "count" in body


@pytest.mark.asyncio
async def test_events_high_impact_endpoint(client):
    resp = await client.get("/api/v1/intelligence/events/high-impact")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert all(e["is_high_impact"] for e in events)


@pytest.mark.asyncio
async def test_events_next_endpoint(client):
    resp = await client.get("/api/v1/intelligence/events/next")
    assert resp.status_code == 200
    body = resp.json()
    assert "event" in body
    assert "minutes_until" in body
    assert body["event"] is not None


@pytest.mark.asyncio
async def test_events_next_high_impact_only(client):
    resp = await client.get("/api/v1/intelligence/events/next?high_impact_only=true")
    assert resp.status_code == 200
    body = resp.json()
    if body["event"]:
        assert body["event"]["is_high_impact"]


# ── Speeches endpoint ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_speeches_endpoint(client):
    resp = await client.get("/api/v1/intelligence/speeches")
    assert resp.status_code == 200
    events = resp.json()["events"]
    # "Fed Chair Powell Speaks" should be classified as a speech
    assert any("Powell" in e["title"] for e in events)
    assert all(e["is_speech"] for e in events)


# ── Canonical model fields ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_canonical_model_fields(client):
    """Every response event must contain all canonical MFIPEvent fields."""
    resp = await client.get("/api/v1/intelligence/calendar")
    assert resp.status_code == 200
    required = {
        "event_id", "provider", "provider_event_id",
        "title", "currency", "country",
        "timestamp_utc", "is_all_day",
        "impact", "is_high_impact", "is_speech", "category",
        "forecast", "previous", "actual",
        "status", "last_updated", "metadata",
    }
    for event in resp.json()["events"]:
        missing = required - set(event.keys())
        assert not missing, f"Event missing fields: {missing}"


@pytest.mark.asyncio
async def test_no_raw_ff_fields_exposed(client):
    """Raw Forex Factory field names must never appear in API responses."""
    resp = await client.get("/api/v1/intelligence/calendar")
    assert resp.status_code == 200
    ff_field_names = {"country_raw", "impact_raw", "time", "url"}   # provider-specific
    for event in resp.json()["events"]:
        for ff_field in ff_field_names:
            assert ff_field not in event, f"Raw FF field '{ff_field}' leaked into response"


@pytest.mark.asyncio
async def test_provider_is_forex_factory(client):
    resp = await client.get("/api/v1/intelligence/calendar")
    assert resp.status_code == 200
    for e in resp.json()["events"]:
        assert e["provider"] == "forex_factory"


# ── Health endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint_structure(client):
    resp = await client.get("/api/v1/intelligence/health")
    assert resp.status_code == 200
    h = resp.json()

    required = {
        "schema_version", "provider",
        "connector_status", "scheduler_running", "started_at", "uptime_s",
        "cache_populated", "calendar_events_total", "calendar_events_high_impact",
        "speeches_cached", "news_items_cached",
        "jobs",
        "calendar_poll_s", "news_poll_s", "sentiment_poll_s", "speeches_poll_s",
    }
    missing = required - set(h.keys())
    assert not missing, f"Health response missing fields: {missing}"


@pytest.mark.asyncio
async def test_health_shows_scheduler_running(client):
    resp = await client.get("/api/v1/intelligence/health")
    assert resp.status_code == 200
    h = resp.json()
    assert h["scheduler_running"] is True
    assert h["connector_status"] in ("ok", "initializing", "degraded")


@pytest.mark.asyncio
async def test_health_cache_populated(client):
    resp = await client.get("/api/v1/intelligence/health")
    assert resp.status_code == 200
    h = resp.json()
    assert h["cache_populated"]["thisweek"] is True
    assert h["calendar_events_total"] == len(MOCK_EVENTS)


@pytest.mark.asyncio
async def test_health_provider_is_forex_factory(client):
    resp = await client.get("/api/v1/intelligence/health")
    assert resp.status_code == 200
    assert resp.json()["provider"] == "forex_factory"


# ── Polling cycle ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_updates_on_new_data():
    """A second poll with new data updates the cache; a 304 leaves it unchanged."""
    from forex_factory_connector.cache.memory_cache import ConnectorCache
    from forex_factory_connector.scheduler.jobs.calendar_job import run_calendar_job
    from forex_factory_connector.cache.etag_store import etag_store

    fresh_cache = ConnectorCache()

    new_events = MOCK_EVENTS + [{
        "title": "US Retail Sales m/m", "country": "USD",
        "date": (datetime.now(timezone.utc) + timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%S"),
        "impact": "High", "forecast": "0.5%", "previous": "0.4%",
    }]
    new_body = json.dumps(new_events).encode()

    with patch(
        "forex_factory_connector.fetcher.cdn_fetcher.fetch_calendar",
        side_effect=AsyncMock(return_value=CDNFetchResult(
            body=new_body, etag='"new-etag"', not_modified=False
        )),
    ):
        with patch("forex_factory_connector.scheduler.jobs.calendar_job.connector_cache", fresh_cache):
            await run_calendar_job("thisweek")
            entry = await fresh_cache.get_calendar("thisweek")
            assert len(entry.events) == len(new_events)

    # Second poll returns 304 — cache should not change
    with patch(
        "forex_factory_connector.fetcher.cdn_fetcher.fetch_calendar",
        side_effect=AsyncMock(return_value=CDNFetchResult(body=None, etag='"new-etag"', not_modified=True)),
    ):
        with patch("forex_factory_connector.scheduler.jobs.calendar_job.connector_cache", fresh_cache):
            await run_calendar_job("thisweek")
            entry = await fresh_cache.get_calendar("thisweek")
            assert len(entry.events) == len(new_events)  # unchanged


# ── Phase 3 stubs ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_news_endpoint_returns_503(client):
    resp = await client.get("/api/v1/intelligence/news")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_sentiment_endpoint_returns_503(client):
    resp = await client.get("/api/v1/intelligence/sentiment")
    assert resp.status_code == 503
