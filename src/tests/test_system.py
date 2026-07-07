"""Tests for /api/v1/system/* endpoints."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    resp = await client.get("/api/v1/system/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_has_status_field(client):
    data = (await client.get("/api/v1/system/health")).json()
    assert "status" in data
    assert data["status"] in ("operational", "degraded")


@pytest.mark.asyncio
async def test_health_has_components(client):
    data = (await client.get("/api/v1/system/health")).json()
    assert "components" in data
    assert isinstance(data["components"], dict)


@pytest.mark.asyncio
async def test_health_has_uptime(client):
    data = (await client.get("/api/v1/system/health")).json()
    assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_status_returns_200(client):
    resp = await client.get("/api/v1/system/status")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_status_engines_online_is_list(client):
    data = (await client.get("/api/v1/system/status")).json()
    assert isinstance(data["engines_online"], list)


@pytest.mark.asyncio
async def test_version_returns_200(client):
    resp = await client.get("/api/v1/system/version")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_version_has_schema_version(client):
    data = (await client.get("/api/v1/system/version")).json()
    assert data["versions"]["decision_schema_version"] == "decision_fusion_v1"


@pytest.mark.asyncio
async def test_logs_returns_200(client):
    resp = await client.get("/api/v1/system/logs")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_logs_response_shape(client):
    data = (await client.get("/api/v1/system/logs")).json()
    assert "logs" in data
    assert "total" in data
