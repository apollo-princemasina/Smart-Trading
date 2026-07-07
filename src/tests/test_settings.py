"""Tests for /api/v1/settings/* endpoints."""
from __future__ import annotations

import pytest

from src.database.models.app_settings import AppSettings


@pytest.mark.asyncio
async def test_settings_list_returns_200(client):
    resp = await client.get("/api/v1/settings")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_settings_list_shape(client):
    data = (await client.get("/api/v1/settings")).json()
    assert "settings" in data
    assert "total" in data
    assert "categories" in data


@pytest.mark.asyncio
async def test_settings_get_nonexistent_returns_404(client):
    resp = await client.get("/api/v1/settings/does_not_exist_xyz")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_settings_update_nonexistent_returns_404(client):
    resp = await client.put("/api/v1/settings/does_not_exist_xyz", json={"value": "123"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_settings_round_trip(client):
    """Insert via service, then read via API."""
    from src.tests.conftest import test_session_factory
    from src.services.settings_service import SettingsService

    svc = SettingsService()
    await svc.set(
        "test_key",
        "hello",
        session_factory=test_session_factory,
        value_type="string",
        category="test",
    )

    resp = await client.get("/api/v1/settings/test_key")
    assert resp.status_code == 200
    data = resp.json()
    assert data["setting"]["key"] == "test_key"
    assert data["setting"]["value"] == "hello"


@pytest.mark.asyncio
async def test_settings_update_value(client):
    from src.tests.conftest import test_session_factory
    from src.services.settings_service import SettingsService

    svc = SettingsService()
    await svc.set(
        "update_test_key",
        "original",
        session_factory=test_session_factory,
        value_type="string",
        category="test",
    )

    resp = await client.put("/api/v1/settings/update_test_key", json={"value": "updated"})
    assert resp.status_code == 200
    assert resp.json()["setting"]["value"] == "updated"
