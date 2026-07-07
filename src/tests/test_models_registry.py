"""Tests for /api/v1/models/* endpoints."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_models_list_returns_200(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_models_list_shape(client):
    data = (await client.get("/api/v1/models")).json()
    assert "models" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_active_model_returns_200(client):
    resp = await client.get("/api/v1/models/active")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_active_model_null_when_none(client):
    data = (await client.get("/api/v1/models/active")).json()
    assert "model" in data


@pytest.mark.asyncio
async def test_register_model_returns_201(client):
    payload = {
        "model_name":    "test_model",
        "model_version": "1.0.0",
        "bundle_path":   "/models/test",
        "feature_count": 247,
        "git_commit":    "abc1234",
    }
    resp = await client.post("/api/v1/models/register", json=payload)
    assert resp.status_code == 201


@pytest.mark.asyncio
async def test_register_activates_model(client):
    payload = {
        "model_name":    "active_test_model",
        "model_version": "2.0.0",
        "bundle_path":   "/models/active_test",
    }
    await client.post("/api/v1/models/register", json=payload)
    active = (await client.get("/api/v1/models/active")).json()
    assert active["model"] is not None
    assert active["model"]["is_active"] is True


@pytest.mark.asyncio
async def test_get_model_by_id(client):
    payload = {
        "model_name":    "lookup_test_model",
        "model_version": "3.0.0",
        "bundle_path":   "/models/lookup",
    }
    reg = (await client.post("/api/v1/models/register", json=payload)).json()
    model_id = reg["model"]["id"]

    resp = await client.get(f"/api/v1/models/{model_id}")
    assert resp.status_code == 200
    assert resp.json()["model"]["id"] == model_id


@pytest.mark.asyncio
async def test_get_model_not_found(client):
    resp = await client.get("/api/v1/models/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
