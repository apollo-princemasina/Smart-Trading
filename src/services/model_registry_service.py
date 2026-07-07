"""ModelRegistryService — track and query model versions."""
from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import select

from src.database.models.model_registry import ModelRegistry


class ModelRegistryService:
    """
    Manages the ModelRegistry table: register new bundles, query active model,
    list history, and deactivate stale versions.
    """

    def __init__(self, session_factory, app_state) -> None:
        self._session_factory = session_factory
        self._state = app_state

    async def get_active(self) -> ModelRegistry | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ModelRegistry)
                .where(ModelRegistry.is_active.is_(True))
                .order_by(ModelRegistry.registered_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def list_all(self, limit: int = 50) -> list[ModelRegistry]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ModelRegistry)
                .order_by(ModelRegistry.registered_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_by_id(self, model_id: str) -> ModelRegistry | None:
        async with self._session_factory() as session:
            return await session.get(ModelRegistry, model_id)

    async def register(
        self,
        *,
        model_name: str,
        model_version: str,
        bundle_path: str,
        git_commit: str | None = None,
        feature_schema_version: str | None = None,
        label_version: str | None = None,
        decision_schema_version: str | None = None,
        pipeline_version: str | None = None,
        training_start: str | None = None,
        training_end: str | None = None,
        training_dataset: str | None = None,
        feature_count: int = 247,
        accuracy: float | None = None,
        precision_buy: float | None = None,
        recall_buy: float | None = None,
        f1_buy: float | None = None,
        precision_sell: float | None = None,
        recall_sell: float | None = None,
        f1_sell: float | None = None,
        notes: str | None = None,
        metrics: dict | None = None,
    ) -> ModelRegistry:
        """Register a new model version and deactivate the previous active one."""
        async with self._session_factory() as session:
            # Deactivate current active
            prev = await session.execute(
                select(ModelRegistry)
                .where(ModelRegistry.is_active.is_(True))
            )
            for row in prev.scalars().all():
                row.is_active = False

            entry = ModelRegistry(
                model_name=model_name,
                model_version=model_version,
                bundle_path=bundle_path,
                git_commit=git_commit,
                feature_schema_version=feature_schema_version,
                label_version=label_version,
                decision_schema_version=decision_schema_version,
                pipeline_version=pipeline_version,
                training_start=training_start,
                training_end=training_end,
                training_dataset=training_dataset,
                feature_count=feature_count,
                accuracy=accuracy,
                precision_buy=precision_buy,
                recall_buy=recall_buy,
                f1_buy=f1_buy,
                precision_sell=precision_sell,
                recall_sell=recall_sell,
                f1_sell=f1_sell,
                is_active=True,
                notes=notes,
                metrics=metrics,
            )
            session.add(entry)
            await session.commit()
            await session.refresh(entry)
            logger.info("ModelRegistry: registered {} v{}", model_name, model_version)
            return entry

    async def register_from_pipeline_manager(self) -> ModelRegistry | None:
        """
        Auto-register the currently loaded model bundle from PipelineManager.

        Skips if this bundle_path is already registered as active.
        """
        try:
            pm = self._state.pipeline_manager
            bundle_path = str(getattr(pm, "_bundle_path", "") or "")
            model_name  = getattr(pm, "_model_name", "unknown")
            feature_count = int(getattr(pm, "_feature_count", 247))

            if not bundle_path:
                return None

            # Skip if already registered
            active = await self.get_active()
            if active and active.bundle_path == bundle_path:
                return active

            return await self.register(
                model_name=model_name,
                model_version=getattr(pm, "_model_version", "unknown"),
                bundle_path=bundle_path,
                feature_count=feature_count,
            )
        except Exception as exc:
            logger.warning("Could not auto-register model: {}", exc)
            return None
