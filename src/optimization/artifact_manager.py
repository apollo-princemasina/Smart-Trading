"""
Artifact Manager
================
Saves and loads a complete, self-contained inference bundle.

Every bundle contains everything needed to reproduce inference in production:
  model.joblib            — fitted estimator
  preprocessing.joblib    — ColumnImputer (or passthrough)
  feature_order.json      — column names in training order
  selected_features.json  — feature metadata (names, dtypes, count)
  schema_version.json     — pipeline schema version
  label_version.json      — label/target metadata
  model_metadata.json     — params, window info, training metadata
  optimization_results.json — best params, n_trials, improvement
  training_metrics.json   — train/val/test metric dictionaries
  inference_config.json   — full configuration for inference
  pipeline_manifest.json  — file list with SHA-256 checksums

InferencePipeline
-----------------
Loading a bundle produces an ``InferencePipeline`` instance that exposes
``predict(df)`` and ``predict_proba(df)`` methods — no manual reconstruction
ever needed.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_FILES = {
    "model.joblib",
    "preprocessing.joblib",
    "feature_order.json",
    "selected_features.json",
    "schema_version.json",
    "label_version.json",
    "model_metadata.json",
    "optimization_results.json",
    "training_metrics.json",
    "inference_config.json",
    "pipeline_manifest.json",
}


# ── Preprocessing artifact ────────────────────────────────────────────────────

class ColumnImputer:
    """Column-wise imputer fitted on training data.

    Stores per-column fill values derived from the training set so that
    inference can reproduce the same imputation without re-fitting.

    Args:
        apply_imputation: If False, transform() is a no-op (used for gradient
                          boosted trees that handle NaN natively).
        strategy:         "median" (default) or "mean".
    """

    def __init__(
        self,
        apply_imputation: bool = True,
        strategy:         str  = "median",
    ) -> None:
        self.apply_imputation = apply_imputation
        self.strategy         = strategy
        self.fill_values:     dict[str, float] = {}
        self.feature_names:   list[str]        = []
        self._fitted          = False

    def fit(self, X: pd.DataFrame) -> "ColumnImputer":
        self.feature_names = list(X.columns)
        for col in self.feature_names:
            if not pd.api.types.is_numeric_dtype(X[col]):
                self.fill_values[col] = 0.0  # non-numeric: skip imputation
                continue
            if self.strategy == "median":
                v = X[col].median()
            else:
                v = X[col].mean()
            try:
                self.fill_values[col] = 0.0 if (v is None or pd.isna(v)) else float(v)
            except (TypeError, ValueError):
                self.fill_values[col] = 0.0
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.apply_imputation or not self.fill_values:
            return X
        X = X.copy()
        for col, val in self.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        """Apply imputation to a numpy array using stored column order."""
        if not self.apply_imputation or not self.fill_values:
            return X
        X = X.copy()
        for j, col in enumerate(self.feature_names):
            val  = self.fill_values.get(col, 0.0)
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = val
        return X

    def to_dict(self) -> dict:
        return {
            "apply_imputation": self.apply_imputation,
            "strategy":         self.strategy,
            "fill_values":      self.fill_values,
            "feature_names":    self.feature_names,
            "fitted":           self._fitted,
        }


# ── Bundle config ─────────────────────────────────────────────────────────────

@dataclass
class BundleConfig:
    """All metadata required to write a complete inference bundle."""
    model_name:              str
    task_type:               str
    target_column:           str
    feature_columns:         list[str]
    n_classes:               Optional[int]
    random_seed:             int
    schema_version:          str
    label_version:           str
    window_number:           int
    train_start:             Optional[str]
    train_end:               Optional[str]
    val_start:               Optional[str]
    val_end:                 Optional[str]
    test_start:              Optional[str]
    test_end:                Optional[str]
    best_params:             dict
    optimization_metric:     str
    n_trials:                int
    optimization_time_s:     float
    best_val_score:          float
    baseline_val_score:      Optional[float]
    training_time_s:         float
    prediction_time_s:       float
    n_train_samples:         int
    n_val_samples:           int
    n_test_samples:          int
    train_metrics:           dict
    val_metrics:             dict
    test_metrics:            dict
    study_name:              str             = ""
    symbol:                  str             = ""


# ── ArtifactManager ───────────────────────────────────────────────────────────

class ArtifactManager:
    """Saves and loads complete inference bundles."""

    # ------------------------------------------------------------------
    @staticmethod
    def save_bundle(
        model:      Any,
        imputer:    ColumnImputer,
        config:     BundleConfig,
        output_dir: Path,
    ) -> Path:
        """Write all bundle artefacts to *output_dir*.

        Args:
            model:      Fitted estimator.
            imputer:    Pre-fitted ColumnImputer.
            config:     Bundle metadata.
            output_dir: Target directory (created if absent).

        Returns:
            Path to the bundle directory (same as output_dir).
        """
        bundle_dir = Path(output_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()

        # ── model ─────────────────────────────────────────────────────────
        joblib.dump(model,   bundle_dir / "model.joblib",          compress=3)
        joblib.dump(imputer, bundle_dir / "preprocessing.joblib",  compress=3)

        # ── feature_order ─────────────────────────────────────────────────
        _write_json(bundle_dir / "feature_order.json", config.feature_columns)

        # ── selected_features ─────────────────────────────────────────────
        _write_json(bundle_dir / "selected_features.json", {
            "feature_count":   len(config.feature_columns),
            "feature_names":   config.feature_columns,
            "schema_version":  config.schema_version,
        })

        # ── schema_version ────────────────────────────────────────────────
        _write_json(bundle_dir / "schema_version.json", {
            "schema_version":   config.schema_version,
            "pipeline_version": "1.0.0",
            "created_at":       now,
        })

        # ── label_version ─────────────────────────────────────────────────
        _write_json(bundle_dir / "label_version.json", {
            "label_version":  config.label_version,
            "target_column":  config.target_column,
            "label_type":     config.task_type,
            "n_classes":      config.n_classes,
        })

        # ── model_metadata ────────────────────────────────────────────────
        model_size = int((bundle_dir / "model.joblib").stat().st_size)
        _write_json(bundle_dir / "model_metadata.json", {
            "model_name":          config.model_name,
            "task_type":           config.task_type,
            "model_params":        config.best_params,
            "window_number":       config.window_number,
            "training_window":     {"start": config.train_start, "end": config.train_end},
            "validation_window":   {"start": config.val_start,   "end": config.val_end},
            "testing_window":      {"start": config.test_start,  "end": config.test_end},
            "training_time_s":     config.training_time_s,
            "prediction_time_s":   config.prediction_time_s,
            "model_size_bytes":    model_size,
            "n_train_samples":     config.n_train_samples,
            "n_val_samples":       config.n_val_samples,
            "n_test_samples":      config.n_test_samples,
            "n_features":          len(config.feature_columns),
            "target_column":       config.target_column,
            "feature_columns":     config.feature_columns,
            "random_seed":         config.random_seed,
            "created_at":          now,
            "symbol":              config.symbol,
        })

        # ── optimization_results ──────────────────────────────────────────
        improvement = None
        if config.baseline_val_score is not None and config.baseline_val_score > 0:
            improvement = round(
                (config.best_val_score - config.baseline_val_score)
                / abs(config.baseline_val_score) * 100, 2
            )
        _write_json(bundle_dir / "optimization_results.json", {
            "study_name":            config.study_name,
            "model_name":            config.model_name,
            "n_trials":              config.n_trials,
            "optimization_time_s":   config.optimization_time_s,
            "best_val_score":        config.best_val_score,
            "baseline_val_score":    config.baseline_val_score,
            "improvement_pct":       improvement,
            "optimization_metric":   config.optimization_metric,
            "best_params":           config.best_params,
            "created_at":            now,
        })

        # ── training_metrics ──────────────────────────────────────────────
        _write_json(bundle_dir / "training_metrics.json", {
            "train": _strip_non_scalars(config.train_metrics),
            "val":   _strip_non_scalars(config.val_metrics),
            "test":  _strip_non_scalars(config.test_metrics),
        })

        # ── inference_config ──────────────────────────────────────────────
        _write_json(bundle_dir / "inference_config.json", {
            "model_name":          config.model_name,
            "task_type":           config.task_type,
            "target_column":       config.target_column,
            "feature_columns":     config.feature_columns,
            "n_features":          len(config.feature_columns),
            "n_classes":           config.n_classes,
            "requires_imputation": imputer.apply_imputation,
            "random_seed":         config.random_seed,
            "schema_version":      config.schema_version,
            "label_version":       config.label_version,
            "created_at":          now,
        })

        # ── pipeline_manifest (last — includes checksums of all files) ────
        manifest = _build_manifest(bundle_dir, config.model_name, now)
        _write_json(bundle_dir / "pipeline_manifest.json", manifest)

        logger.info(
            "Bundle saved → %s (%d files, model=%.1f KB)",
            bundle_dir, len(manifest["files"]),
            model_size / 1024,
        )
        return bundle_dir

    # ------------------------------------------------------------------
    @staticmethod
    def load_bundle(bundle_dir: Path) -> "InferencePipeline":
        """Load a bundle and return a ready-to-use InferencePipeline."""
        return InferencePipeline(bundle_dir)

    # ------------------------------------------------------------------
    @staticmethod
    def verify_bundle(bundle_dir: Path) -> tuple[bool, list[str]]:
        """Check that all required files are present.

        Returns:
            (is_complete, list_of_missing_files)
        """
        bundle_dir = Path(bundle_dir)
        missing = [f for f in _REQUIRED_FILES if not (bundle_dir / f).exists()]
        return len(missing) == 0, missing

    # ------------------------------------------------------------------
    @staticmethod
    def copy_to_best(
        source_bundle_dir: Path,
        best_dir:          Path,
    ) -> Path:
        """Copy a bundle to ``best_dir`` as the canonical best model.

        Overwrites any existing content in *best_dir*.
        Returns *best_dir*.
        """
        import shutil
        best_dir = Path(best_dir)
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(source_bundle_dir, best_dir)
        logger.info("Best model bundle → %s", best_dir)
        return best_dir


# ── InferencePipeline ─────────────────────────────────────────────────────────

class InferencePipeline:
    """Self-contained inference pipeline loaded from a bundle directory.

    Loading the bundle automatically restores model, preprocessing,
    feature ordering, and configuration.  No manual reconstruction needed.

    Usage
    -----
        pipe = InferencePipeline("models/best_model")
        predictions = pipe.predict(df)        # → numpy array
        probabilities = pipe.predict_proba(df) # → numpy array or None
    """

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = Path(bundle_dir)
        self._model:          Any           = None
        self._preprocessing:  ColumnImputer = None
        self._inference_cfg:  dict          = {}
        self._feature_order:  list[str]     = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        ok, missing = ArtifactManager.verify_bundle(self.bundle_dir)
        if not ok:
            raise FileNotFoundError(
                f"Bundle incomplete — missing files: {missing}. "
                f"Bundle directory: {self.bundle_dir}"
            )
        self._model         = joblib.load(self.bundle_dir / "model.joblib")
        self._preprocessing = joblib.load(self.bundle_dir / "preprocessing.joblib")
        self._inference_cfg = json.loads(
            (self.bundle_dir / "inference_config.json").read_text(encoding="utf-8")
        )
        self._feature_order = json.loads(
            (self.bundle_dir / "feature_order.json").read_text(encoding="utf-8")
        )

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._inference_cfg.get("model_name", "")

    @property
    def task_type(self) -> str:
        return self._inference_cfg.get("task_type", "classification")

    @property
    def target_column(self) -> str:
        return self._inference_cfg.get("target_column", "")

    @property
    def feature_columns(self) -> list[str]:
        return self._inference_cfg.get("feature_columns", self._feature_order)

    @property
    def n_features(self) -> int:
        return self._inference_cfg.get("n_features", len(self._feature_order))

    @property
    def requires_imputation(self) -> bool:
        return self._inference_cfg.get("requires_imputation", False)

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return class predictions for rows in *df*."""
        X = self._prepare(df)
        return self._model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Return class probabilities, or None if the model cannot."""
        if not hasattr(self._model, "predict_proba"):
            return None
        X = self._prepare(df)
        return self._model.predict_proba(X)

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self._feature_order]
        X = self._preprocessing.transform(X)
        return X.to_numpy(dtype=float, na_value=np.nan)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (f != f) else f   # NaN → null
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _strip_non_scalars(metrics: dict) -> dict:
    """Remove list/dict values (e.g. confusion_matrix) for clean JSON."""
    return {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(bundle_dir: Path, model_name: str, created_at: str) -> dict:
    files = []
    for filename in sorted(f.name for f in bundle_dir.iterdir() if f.is_file()):
        p = bundle_dir / filename
        files.append({
            "filename":    filename,
            "size_bytes":  int(p.stat().st_size),
            "sha256":      _sha256(p),
            "required":    filename in _REQUIRED_FILES,
            "description": _FILE_DESCRIPTIONS.get(filename, ""),
        })
    return {
        "manifest_version":  "1.0.0",
        "bundle_type":       "inference_pipeline",
        "model_name":        model_name,
        "created_at":        created_at,
        "required_files":    sorted(_REQUIRED_FILES),
        "files":             files,
        # pipeline_manifest.json is written AFTER this dict is built, so exclude it
        "is_complete":       all(
            (bundle_dir / f).exists()
            for f in _REQUIRED_FILES if f != "pipeline_manifest.json"
        ),
    }


_FILE_DESCRIPTIONS: dict[str, str] = {
    "model.joblib":             "Fitted estimator (joblib-serialised)",
    "preprocessing.joblib":     "ColumnImputer preprocessing artifact",
    "feature_order.json":       "Ordered list of feature column names",
    "selected_features.json":   "Feature metadata (names, count, schema version)",
    "schema_version.json":      "Pipeline schema version information",
    "label_version.json":       "Label and target metadata",
    "model_metadata.json":      "Model params, window info, training metadata",
    "optimization_results.json":"Best hyperparameters, n_trials, improvement",
    "training_metrics.json":    "Train/val/test metric dictionaries",
    "inference_config.json":    "Complete inference configuration",
    "pipeline_manifest.json":   "Manifest with SHA-256 checksums for all files",
}
