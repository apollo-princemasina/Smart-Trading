"""Tests for /api/v1/history/* endpoints."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_prediction_history_returns_200(client):
    resp = await client.get("/api/v1/history/predictions")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_prediction_history_shape(client):
    data = (await client.get("/api/v1/history/predictions")).json()
    assert "predictions" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


@pytest.mark.asyncio
async def test_prediction_history_pagination(client):
    resp = await client.get("/api/v1/history/predictions?page=1&page_size=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 1
    assert data["page_size"] == 5


@pytest.mark.asyncio
async def test_decision_history_returns_200(client):
    resp = await client.get("/api/v1/history/decisions")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_decision_history_shape(client):
    data = (await client.get("/api/v1/history/decisions")).json()
    assert "decisions" in data
    assert "total" in data
    assert "page" in data


@pytest.mark.asyncio
async def test_combined_history_returns_200(client):
    resp = await client.get("/api/v1/history/combined")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_combined_history_shape(client):
    data = (await client.get("/api/v1/history/combined")).json()
    assert "items" in data
    assert "page" in data
    assert "page_size" in data


@pytest.mark.asyncio
async def test_decision_history_filter_by_recommendation(client):
    resp = await client.get("/api/v1/history/decisions?recommendation=BUY")
    assert resp.status_code == 200
