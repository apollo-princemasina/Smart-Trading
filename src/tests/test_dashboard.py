"""Tests for GET /api/v1/dashboard."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_dashboard_returns_200(client):
    resp = await client.get("/api/v1/dashboard")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_dashboard_has_required_sections(client):
    data = (await client.get("/api/v1/dashboard")).json()
    assert "decision" in data
    assert "latest_prediction" in data
    assert "system_summary" in data


@pytest.mark.asyncio
async def test_dashboard_system_summary_has_uptime(client):
    data = (await client.get("/api/v1/dashboard")).json()
    ss = data.get("system_summary", {})
    assert "uptime_seconds" in ss
    assert ss["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_dashboard_decision_is_null_when_empty(client):
    data = (await client.get("/api/v1/dashboard")).json()
    # No decision has been produced in test env — should be null, not crash
    assert data["decision"] is None
