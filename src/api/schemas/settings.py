"""Pydantic schemas for settings endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


class SettingOut(BaseModel):
    key:         str
    value:       Any           # redacted to "***" for is_secret=True
    value_type:  str
    category:    str
    description: str | None
    is_secret:   bool
    updated_at:  Any


class SettingUpdateRequest(BaseModel):
    value: Any

    @field_validator("value", mode="before")
    @classmethod
    def not_none(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("value cannot be null")
        return v


class SettingsListResponse(BaseModel):
    settings:   list[SettingOut]
    total:      int
    categories: list[str]


class SettingResponse(BaseModel):
    setting: SettingOut
