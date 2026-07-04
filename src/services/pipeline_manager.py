"""Pipeline Manager — loads the trained model bundle once at startup.

The bundle is NEVER reloaded during the inference loop.
Use reload() only via an admin API call after a new bundle is deployed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from src.api.core.config import settings


class PipelineManager:
    """Holds the InferencePipeline singleton."""

    def __init__(self, bundle_dir: Optional[Path] = None) -> None:
        self._bundle_dir  = Path(bundle_dir or settings.MODEL_BUNDLE_DIR)
        self._pipeline    = None
        self._loaded_at: Optional[datetime] = None
        self._feature_count: int = 0
        self._model_name:    str = ""

    # ── Public interface ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the bundle from disk.  Raises on failure (blocks startup)."""
        from src.optimization.artifact_manager import InferencePipeline

        logger.info("Loading model bundle from {}", self._bundle_dir)
        self._pipeline    = InferencePipeline(self._bundle_dir)
        self._loaded_at   = datetime.now(timezone.utc)
        self._feature_count = len(self._pipeline._feature_order)
        self._model_name  = self._bundle_dir.name
        logger.info(
            "Model bundle loaded — name={} features={}",
            self._model_name, self._feature_count,
        )

    def reload(self) -> None:
        """Hot-reload the bundle (e.g. after a new model is deployed)."""
        logger.warning("Hot-reloading model bundle...")
        self.load()

    @property
    def pipeline(self):
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded — call load() first")
        return self._pipeline

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    @property
    def feature_count(self) -> int:
        return self._feature_count

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def bundle_dir(self) -> Path:
        return self._bundle_dir
