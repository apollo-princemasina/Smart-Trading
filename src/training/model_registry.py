"""
Model Registry
==============
Saves trained estimators to disk and stores rich metadata alongside each
model file.

Output layout for one model in one window
------------------------------------------
    models/
        window_001/
            xgboost.joblib
            xgboost_metadata.json
            lightgbm.joblib
            lightgbm_metadata.json
            ...

The metadata JSON captures everything needed to reproduce a model's evaluation
results without re-running training: window dates, metric values, feature list,
random seed, model size, and timing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib

from .trainer import ModelWindowResult

logger = logging.getLogger(__name__)


# ── Metadata dataclass ────────────────────────────────────────────────────────

@dataclass
class ModelMeta:
    """Full provenance record for one saved model."""
    model_name:              str
    task_type:               str
    window_number:           int
    target_column:           str
    feature_columns:         list[str]
    feature_count:           int
    n_train_samples:         int
    n_val_samples:           int
    n_test_samples:          int
    training_time_seconds:   float
    prediction_time_seconds: float
    model_path:              str
    model_size_bytes:        int
    random_seed:             int
    schema_version:          str
    training_window:         dict     # {start, end}
    validation_window:       dict
    testing_window:          dict
    train_metrics:           dict
    val_metrics:             dict
    test_metrics:            dict
    n_classes:               Optional[int] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=_json_default),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> "ModelMeta":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _json_default(obj):
    """Fallback JSON serialiser for numpy scalars etc."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ── Registry ──────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Save / load trained models and their metadata."""

    @staticmethod
    def save(
        model:             Any,
        result:            ModelWindowResult,
        output_dir:        Path,
        random_seed:       int  = 42,
        schema_version:    str  = "1.0.0",
        train_start:       Optional[str] = None,
        train_end:         Optional[str] = None,
        val_start:         Optional[str] = None,
        val_end:           Optional[str] = None,
        test_start:        Optional[str] = None,
        test_end:          Optional[str] = None,
    ) -> Path:
        """Persist *model* to disk and write its metadata JSON.

        Args:
            model:          Fitted estimator.
            result:         ModelWindowResult produced by Trainer.
            output_dir:     Window-specific directory (created if absent).
            random_seed:    Seed used during training (echoed in metadata).
            schema_version: Pipeline schema version.
            train_start/end, val_start/end, test_start/end:
                            ISO-format window boundary timestamps.

        Returns:
            Path to the saved .joblib file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{result.model_name}.joblib"
        joblib.dump(model, model_path, compress=3)
        model_size = int(model_path.stat().st_size)

        meta = ModelMeta(
            model_name              = result.model_name,
            task_type               = result.task_type,
            window_number           = result.window_number,
            target_column           = result.target_column,
            feature_columns         = result.feature_columns,
            feature_count           = result.n_features,
            n_train_samples         = result.n_train,
            n_val_samples           = result.n_val,
            n_test_samples          = result.n_test,
            training_time_seconds   = result.training_time_seconds,
            prediction_time_seconds = result.prediction_time_seconds,
            model_path              = str(model_path),
            model_size_bytes        = model_size,
            random_seed             = random_seed,
            schema_version          = schema_version,
            training_window   = {"start": train_start, "end": train_end},
            validation_window = {"start": val_start,   "end": val_end},
            testing_window    = {"start": test_start,  "end": test_end},
            train_metrics     = result.train_metrics,
            val_metrics       = result.val_metrics,
            test_metrics      = result.test_metrics,
            n_classes         = result.n_classes,
        )
        meta_path = output_dir / f"{result.model_name}_metadata.json"
        meta.to_json(meta_path)

        logger.info(
            "Saved %s → %s (%.1f KB)",
            result.model_name, model_path, model_size / 1024,
        )
        return model_path

    @staticmethod
    def load(path: Path) -> Any:
        """Load a fitted estimator from a .joblib file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    @staticmethod
    def load_meta(path: Path) -> ModelMeta:
        """Load metadata from a _metadata.json file."""
        return ModelMeta.from_json(path)

    @staticmethod
    def list_models(base_dir: Path) -> list[ModelMeta]:
        """Discover and load all *_metadata.json files under *base_dir*.

        Returns a list sorted by (window_number, model_name).
        """
        base_dir = Path(base_dir)
        meta_files = sorted(base_dir.rglob("*_metadata.json"))
        metas: list[ModelMeta] = []
        for p in meta_files:
            try:
                metas.append(ModelMeta.from_json(p))
            except Exception as exc:
                logger.warning("Could not load metadata %s: %s", p, exc)
        return sorted(metas, key=lambda m: (m.window_number, m.model_name))
