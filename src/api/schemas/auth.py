"""Pydantic schemas for authentication endpoints."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int   # seconds until access token expires


class RefreshRequest(BaseModel):
    refresh_token: str


class UserOut(BaseModel):
    id:                str
    email:             str
    username:          str
    role:              str
    subscription_tier: str
    is_active:         bool
    last_login:        str | None


class TokenPayload(BaseModel):
    sub:  str          # user ID
    type: str = "access"   # access | refresh
    exp:  int | None = None
