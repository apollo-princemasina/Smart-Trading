"""Settings endpoints — runtime key-value configuration."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.settings import (
    SettingOut,
    SettingResponse,
    SettingUpdateRequest,
    SettingsListResponse,
)

router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get(
    "",
    response_model=SettingsListResponse,
    summary="List all application settings",
)
async def list_settings(request: Request) -> SettingsListResponse:
    from src.database.session import async_session_factory

    svc = request.app.state.settings_service
    rows = await svc.get_all_rows(async_session_factory)

    items = [
        SettingOut(
            key=r.key,
            value="***" if r.is_secret else r.value,
            value_type=r.value_type,
            category=r.category,
            description=r.description,
            is_secret=r.is_secret,
            updated_at=r.updated_at,
        )
        for r in rows
    ]
    categories = sorted({r.category for r in rows})
    return SettingsListResponse(settings=items, total=len(items), categories=categories)


@router.get(
    "/{key}",
    response_model=SettingResponse,
    summary="Get a single setting by key",
)
async def get_setting(key: str, request: Request) -> SettingResponse:
    from src.database.session import async_session_factory
    from src.database.repositories.settings_repo import SettingsRepository

    async with async_session_factory() as session:
        repo = SettingsRepository(session)
        row = await repo.get_by_key(key)

    if row is None:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")

    return SettingResponse(
        setting=SettingOut(
            key=row.key,
            value="***" if row.is_secret else row.value,
            value_type=row.value_type,
            category=row.category,
            description=row.description,
            is_secret=row.is_secret,
            updated_at=row.updated_at,
        )
    )


@router.put(
    "/{key}",
    response_model=SettingResponse,
    summary="Update a setting value at runtime",
)
async def update_setting(
    key: str, body: SettingUpdateRequest, request: Request
) -> SettingResponse:
    from src.database.session import async_session_factory
    from src.database.repositories.settings_repo import SettingsRepository

    # Check exists
    async with async_session_factory() as session:
        repo = SettingsRepository(session)
        row = await repo.get_by_key(key)

    if row is None:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")

    svc = request.app.state.settings_service
    await svc.set(
        key,
        body.value,
        session_factory=async_session_factory,
        value_type=row.value_type,
        category=row.category,
        description=row.description,
        is_secret=row.is_secret,
    )

    # Re-fetch for response
    async with async_session_factory() as session:
        repo = SettingsRepository(session)
        updated = await repo.get_by_key(key)

    return SettingResponse(
        setting=SettingOut(
            key=updated.key,
            value="***" if updated.is_secret else updated.value,
            value_type=updated.value_type,
            category=updated.category,
            description=updated.description,
            is_secret=updated.is_secret,
            updated_at=updated.updated_at,
        )
    )
