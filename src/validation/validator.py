"""
Window Validator
================
Evaluates a single optimized model bundle on a single walk-forward window's
test split.

Read-only contract
------------------
  * Never calls model.fit() or modifies any model weight.
  * Only reads bundle files and test parquet.
  * Returns a frozen WindowValidationResult.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_trading_metrics,
)

logger = logging.getLogger(__name__)
_NaN = float("nan")

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class WindowValidationResult:
    """All validation outputs for one model evaluated on one window's test split.

    Attributes:
        window_number:       Walk-forward window index.
        model_name:          Name of the evaluated model.
        task_type:           "classification" or "regression".
        classification_metrics: Full classification metric dict (or empty for regression).
        regression_metrics:  Full regression metric dict (or empty for classification).
        trading_metrics:     Extended trading metric dict.
        combined_metrics:    Union of all scalar metrics for downstream aggregation.
        inference_time_s:    Wall-clock seconds for predict() call.
        n_test:              Number of test samples.
        y_true:              Ground-truth labels (stored for regime analysis).
        y_pred:              Model predictions.
        y_prob:              Class probabilities (None for regression).
        bundle_dir:          Path to the model bundle that was evaluated.
        bundle_val_metrics:  Val metrics stored inside the bundle (for overfitting detection).
        bundle_train_metrics: Train metrics stored inside the bundle.
        error:               Non-None if evaluation failed; metrics are empty.
    """
    window_number:          int
    model_name:             str
    task_type:              str
    classification_metrics: dict                 = field(default_factory=dict)
    regression_metrics:     dict                 = field(default_factory=dict)
    trading_metrics:        dict                 = field(default_factory=dict)
    combined_metrics:       dict                 = field(default_factory=dict)
    inference_time_s:       float                = 0.0
    n_test:                 int                  = 0
    y_true:                 Optional[np.ndarray] = field(default=None, repr=False)
    y_pred:                 Optional[np.ndarray] = field(default=None, repr=False)
    y_prob:                 Optional[np.ndarray] = field(default=None, repr=False)
    bundle_dir:             Optional[Path]       = None
    bundle_val_metrics:     Optional[dict]       = None
    bundle_train_metrics:   Optional[dict]       = None
    error:                  Optional[str]        = None


# ── Validator ─────────────────────────────────────────────────────────────────

class WindowValidator:
    """Evaluate one inference bundle on one test split.

    Usage::

        validator = WindowValidator()
        result = validator.validate(
            bundle_dir    = Path("models/window_001/xgboost/bundle"),
            test_df       = pd.read_parquet("data/ml/windows/window_001/test.parquet"),
            target_column = "direction_1b",
            window_number = 1,
        )
    """

    def validate(
        self,
        bundle_dir:    Path,
        test_df:       pd.DataFrame,
        target_column: str,
        window_number: int,
    ) -> WindowValidationResult:
        """Evaluate a bundle on a test split.

        Args:
            bundle_dir:    Path to the inference bundle directory.
            test_df:       Test split DataFrame (must contain target_column).
            target_column: Name of the label column.
            window_number: Window index for identification.

        Returns:
            WindowValidationResult with all metrics or an error record.
        """
        bundle_dir = Path(bundle_dir)

        # ── Load bundle metadata ───────────────────────────────────────────────
        try:
            model_name, task_type = _load_bundle_meta(bundle_dir)
        except Exception as exc:
            return _error_result(window_number, "unknown", str(exc), bundle_dir)

        # ── Extract features and target ───────────────────────────────────────
        if target_column not in test_df.columns:
            return _error_result(
                window_number, model_name,
                f"target column '{target_column}' not found in test split",
                bundle_dir,
            )

        y_true    = test_df[target_column].to_numpy()
        X_test_df = test_df.drop(columns=[target_column])

        # ── Load InferencePipeline ─────────────────────────────────────────────
        try:
            from src.optimization.artifact_manager import ArtifactManager, InferencePipeline
            ok, missing = ArtifactManager.verify_bundle(bundle_dir)
            if not ok:
                return _error_result(
                    window_number, model_name,
                    f"Incomplete bundle — missing: {missing}", bundle_dir,
                )
            pipeline = InferencePipeline(bundle_dir)
        except Exception as exc:
            return _error_result(window_number, model_name, str(exc), bundle_dir)

        # ── Inference (read-only) ──────────────────────────────────────────────
        try:
            t0            = time.monotonic()
            y_pred        = pipeline.predict(X_test_df)
            y_prob        = pipeline.predict_proba(X_test_df)
            inference_time = time.monotonic() - t0
        except Exception as exc:
            return _error_result(window_number, model_name, f"Inference failed: {exc}", bundle_dir)

        # ── Compute metrics ────────────────────────────────────────────────────
        clf_metrics   = {}
        reg_metrics   = {}
        trade_metrics = {}

        if task_type == "classification":
            try:
                clf_metrics   = compute_classification_metrics(y_true, y_pred, y_prob)
            except Exception as exc:
                logger.warning("Classification metrics failed w%d/%s: %s",
                               window_number, model_name, exc)
            try:
                trade_metrics = compute_trading_metrics(y_true, y_pred, y_prob)
            except Exception as exc:
                logger.warning("Trading metrics failed w%d/%s: %s",
                               window_number, model_name, exc)
        else:
            try:
                reg_metrics = compute_regression_metrics(y_true, y_pred)
            except Exception as exc:
                logger.warning("Regression metrics failed w%d/%s: %s",
                               window_number, model_name, exc)

        # ── Combined scalar metrics dict ───────────────────────────────────────
        combined: dict = {}
        for m in (clf_metrics, reg_metrics, trade_metrics):
            combined.update(
                {k: v for k, v in m.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
            )

        # ── Load bundle's stored train/val metrics for overfitting detection ───
        bundle_train_metrics, bundle_val_metrics = _load_stored_metrics(bundle_dir)

        logger.info(
            "Validated w%03d / %-15s | n=%d | time=%.3fs | f1=%s",
            window_number, model_name, len(y_true), inference_time,
            f"{clf_metrics.get('f1', _NaN):.4f}" if clf_metrics else "n/a",
        )

        return WindowValidationResult(
            window_number          = window_number,
            model_name             = model_name,
            task_type              = task_type,
            classification_metrics = clf_metrics,
            regression_metrics     = reg_metrics,
            trading_metrics        = trade_metrics,
            combined_metrics       = combined,
            inference_time_s       = inference_time,
            n_test                 = int(len(y_true)),
            y_true                 = y_true,
            y_pred                 = y_pred,
            y_prob                 = y_prob,
            bundle_dir             = bundle_dir,
            bundle_val_metrics     = bundle_val_metrics,
            bundle_train_metrics   = bundle_train_metrics,
            error                  = None,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_bundle_meta(bundle_dir: Path) -> tuple[str, str]:
    """Return (model_name, task_type) from inference_config.json."""
    cfg_path = bundle_dir / "inference_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"inference_config.json not found in {bundle_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg["model_name"], cfg["task_type"]


def _load_stored_metrics(bundle_dir: Path) -> tuple[Optional[dict], Optional[dict]]:
    """Load train/val metrics from the bundle's training_metrics.json."""
    metrics_path = bundle_dir / "training_metrics.json"
    if not metrics_path.exists():
        return None, None
    try:
        raw   = json.loads(metrics_path.read_text(encoding="utf-8"))
        train = raw.get("train") or {}
        val   = raw.get("val")   or {}
        return dict(train), dict(val)
    except Exception:
        return None, None


def _error_result(
    window_number: int,
    model_name:    str,
    error:         str,
    bundle_dir:    Optional[Path] = None,
) -> WindowValidationResult:
    logger.error("Validation error w%03d / %s: %s", window_number, model_name, error)
    return WindowValidationResult(
        window_number = window_number,
        model_name    = model_name,
        task_type     = "unknown",
        error         = error,
        bundle_dir    = bundle_dir,
    )
