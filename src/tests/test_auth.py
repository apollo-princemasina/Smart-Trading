"""Tests for /api/v1/auth/* endpoints and JWT utilities."""
from __future__ import annotations

import pytest

from src.auth.jwt_utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


# ── JWT utilities ──────────────────────────────────────────────────────────────

def test_hash_and_verify_password():
    hashed = hash_password("secret123")
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong",     hashed) is False


def test_create_and_decode_access_token():
    token = create_access_token("user-001")
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "user-001"
    assert payload["type"] == "access"


def test_create_and_decode_refresh_token():
    token = create_refresh_token("user-001")
    payload = decode_token(token)
    assert payload is not None
    assert payload["type"] == "refresh"


def test_decode_garbage_returns_none():
    assert decode_token("not.a.real.token") is None


def test_decode_expired_returns_none():
    token = create_access_token("user-001", expires_minutes=-1)
    assert decode_token(token) is None


# ── API endpoints ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_login_with_no_user_returns_401(client):
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "noone@example.com", "password": "pass"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_without_token_returns_401(client):
    resp = await client.get("/api/v1/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_with_invalid_token_returns_401(client):
    resp = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalidtoken"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh_with_invalid_token_returns_401(client):
    resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "bad.token.here"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_with_valid_user(client):
    """Create a user, then log in and receive tokens."""
    from src.tests.conftest import test_session_factory
    from src.database.models.user import User
    from src.auth.jwt_utils import hash_password

    async with test_session_factory() as session:
        user = User(
            email="test@mfip.io",
            username="testuser",
            hashed_password=hash_password("testpass"),
            role="analyst",
        )
        session.add(user)
        await session.commit()

    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "test@mfip.io", "password": "testpass"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_me_with_valid_token(client):
    from src.tests.conftest import test_session_factory
    from src.database.models.user import User
    from src.auth.jwt_utils import hash_password, create_access_token

    async with test_session_factory() as session:
        user = User(
            email="me_test@mfip.io",
            username="me_testuser",
            hashed_password=hash_password("pw"),
            role="viewer",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        uid = user.id

    token = create_access_token(uid, extra_claims={"role": "viewer"})
    resp = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "viewer"
