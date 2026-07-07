"""Auth dependency injection — optional and required user guards."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.jwt_utils import decode_token

_bearer = HTTPBearer(auto_error=False)


def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
):
    """
    Return the token payload if a valid Bearer token is provided, else None.

    Use this on endpoints that should work both authenticated and anonymous.
    """
    if credentials is None:
        return None
    payload = decode_token(credentials.credentials)
    return payload  # dict with sub, type, exp — or None if invalid


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
):
    """
    Require a valid Bearer token.

    Raises HTTP 401 if missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def require_role(*roles: str):
    """
    Factory for role-based route protection.

    Usage:
        @router.post("/admin/action")
        async def action(user=Depends(require_role("admin"))):
            ...
    """
    def _check(user=Depends(get_current_user)):
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {roles}",
            )
        return user
    return _check
