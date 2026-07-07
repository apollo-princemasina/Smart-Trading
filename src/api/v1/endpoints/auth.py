"""Auth endpoints — login, refresh, current user."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from src.api.schemas.auth import LoginRequest, RefreshRequest, TokenResponse, UserOut
from src.auth.dependencies import get_current_user
from src.auth.jwt_utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from src.api.core.config import settings

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for JWT tokens",
)
async def login(body: LoginRequest) -> TokenResponse:
    from src.database.session import async_session_factory
    from src.database.models.user import User

    async with async_session_factory() as session:
        result = await session.execute(
            select(User).where(User.email == body.email, User.is_active.is_(True))
        )
        user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last_login
    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user.id))
        db_user = result.scalar_one()
        db_user.last_login = datetime.now(timezone.utc)
        await session.commit()

    extra = {"role": user.role, "tier": user.subscription_tier}
    access  = create_access_token(user.id, extra_claims=extra)
    refresh = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new access token",
)
async def refresh_token(body: RefreshRequest) -> TokenResponse:
    payload = decode_token(body.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user_id = payload.get("sub")
    from src.database.session import async_session_factory
    from src.database.models.user import User

    async with async_session_factory() as session:
        user = await session.get(User, user_id)

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    extra  = {"role": user.role, "tier": user.subscription_tier}
    access = create_access_token(user.id, extra_claims=extra)
    new_refresh = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get(
    "/me",
    response_model=UserOut,
    summary="Return the authenticated user's profile",
)
async def me(token_payload: dict = Depends(get_current_user)) -> UserOut:
    from src.database.session import async_session_factory
    from src.database.models.user import User

    user_id = token_payload.get("sub")
    async with async_session_factory() as session:
        user = await session.get(User, user_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return UserOut(
        id=str(user.id),
        email=user.email,
        username=user.username,
        role=user.role,
        subscription_tier=user.subscription_tier,
        is_active=user.is_active,
        last_login=user.last_login.isoformat() if user.last_login else None,
    )
