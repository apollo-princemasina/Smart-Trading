"""
Trainer
=======
Trains a single estimator on one walk-forward window and returns fully
computed metrics for every split (train / validation / test).

Responsibilities
----------------
* Extract X / y from DataFrames.
* Auto-detect task type when config.task_type == "auto".
* Impute NaN features for sklearn models that cannot handle them natively
  (RandomForest, ExtraTrees).  Gradient-boosted trees (XGBoost, LightGBM,
  CatBoost) receive raw NaN values, which they handle as a separate branch.
* Time the fit and predict calls separately.
* Return a ModelWindowResult dataclass with all metrics and metadata.
"""
from __future__ import annotations

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
    detect_task_type,
)
from .model_factory import SKLEARN_MODELS

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    """Configuration passed to Trainer.train_window()."""
    target_column:          str
    feature_columns:        Optional[list[str]] = None  # None → all non-target cols
    task_type:              str                 = "auto"   # auto / classification / regression
    random_seed:            int                 = 42
    n_jobs:                 int                 = -1
    compute_trading_metrics: bool               = True
    metrics_average:        str                 = "macro"  # for classification


@dataclass
class ModelWindowResult:
    """All outputs for one model trained on one walk-forward window."""
    model_name:              str
    window_number:           int
    task_type:               str
    train_metrics:           dict
    val_metrics:             dict
    test_metrics:            dict
    training_time_seconds:   float
    prediction_time_seconds: float   # mean per-split prediction time
    n_train:                 int
    n_val:                   int
    n_test:                  int
    n_features:              int
    target_column:           str
    feature_columns:         list[str]
    n_classes:               Optional[int]
    model_path:              Optional[Path] = None  # filled in by ModelRegistry.save()


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """Fits an estimator and evaluates it on three splits."""

    def train_window(
        self,
        model:         Any,
        model_name:    str,
        train_df:      pd.DataFrame,
        val_df:        pd.DataFrame,
        test_df:       pd.DataFrame,
        config:        TrainerConfig,
        window_number: int,
    ) -> ModelWindowResult:
        """Train *model* on *train_df* and evaluate on all three splits.

        Args:
            model:         Freshly created (unfitted) estimator.
            model_name:    String identifier (used to decide NaN handling).
            train_df:      Training split DataFrame with DatetimeIndex.
            val_df:        Validation split DataFrame.
            test_df:       Test split DataFrame.
            config:        Trainer configuration.
            window_number: Walk-forward window index (for logging / metadata).

        Returns:
            ModelWindowResult with metrics, timing, and split sizes.
        """
        target = config.target_column

        # ── Feature column selection ──────────────────────────────────────────
        feat_cols = config.feature_columns or [c for c in train_df.columns if c != target]

        # ── Extract arrays ────────────────────────────────────────────────────
        X_train, y_train = self._extract(train_df, feat_cols, target)
        X_val,   y_val   = self._extract(val_df,   feat_cols, target)
        X_test,  y_test  = self._extract(test_df,  feat_cols, target)

        # ── Task type detection ───────────────────────────────────────────────
        task_type = config.task_type
        if task_type == "auto":
            task_type = detect_task_type(pd.Series(y_train))

        # ── NaN imputation (sklearn models only) ──────────────────────────────
        if model_name in SKLEARN_MODELS:
            X_train, X_val, X_test = _impute(X_train, X_val, X_test)

        # ── Fit ───────────────────────────────────────────────────────────────
        t0 = time.monotonic()
        model.fit(X_train, y_train)
        training_time = time.monotonic() - t0

        # ── Predict ───────────────────────────────────────────────────────────
        pred_times: list[float] = []

        def _predict(X: np.ndarray):
            t = time.monotonic()
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
            pred_times.append(time.monotonic() - t)
            return y_pred, y_prob

        y_pred_train, y_prob_train = _predict(X_train)
        y_pred_val,   y_prob_val   = _predict(X_val)
        y_pred_test,  y_prob_test  = _predict(X_test)
        mean_pred_time = float(np.mean(pred_times))

        # ── Metrics ───────────────────────────────────────────────────────────
        if task_type == "classification":
            avg = config.metrics_average
            train_m = {
                **compute_classification_metrics(y_train, y_pred_train, y_prob_train, avg),
                **(compute_trading_metrics(y_train, y_pred_train, y_prob_train)
                   if config.compute_trading_metrics else {}),
            }
            val_m = {
                **compute_classification_metrics(y_val, y_pred_val, y_prob_val, avg),
                **(compute_trading_metrics(y_val, y_pred_val, y_prob_val)
                   if config.compute_trading_metrics else {}),
            }
            test_m = {
                **compute_classification_metrics(y_test, y_pred_test, y_prob_test, avg),
                **(compute_trading_metrics(y_test, y_pred_test, y_prob_test)
                   if config.compute_trading_metrics else {}),
            }
            n_classes = int(len(np.unique(y_train)))
        else:
            train_m = compute_regression_metrics(y_train, y_pred_train)
            val_m   = compute_regression_metrics(y_val,   y_pred_val)
            test_m  = compute_regression_metrics(y_test,  y_pred_test)
            n_classes = None

        logger.debug(
            "Window %03d | %s | task=%s | train=%d val=%d test=%d | "
            "fit=%.2fs pred=%.3fs",
            window_number, model_name, task_type,
            len(X_train), len(X_val), len(X_test),
            training_time, mean_pred_time,
        )

        return ModelWindowResult(
            model_name              = model_name,
            window_number           = window_number,
            task_type               = task_type,
            train_metrics           = train_m,
            val_metrics             = val_m,
            test_metrics            = test_m,
            training_time_seconds   = float(training_time),
            prediction_time_seconds = mean_pred_time,
            n_train                 = len(X_train),
            n_val                   = len(X_val),
            n_test                  = len(X_test),
            n_features              = len(feat_cols),
            target_column           = target,
            feature_columns         = list(feat_cols),
            n_classes               = n_classes,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract(df: pd.DataFrame, feat_cols: list[str], target: str):
        X = df[feat_cols].to_numpy(dtype=float, na_value=np.nan)
        y = df[target].to_numpy()
        return X, y


# ── NaN imputation ────────────────────────────────────────────────────────────

def _impute(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill NaN with column medians computed on the training set.

    Column medians are never derived from val/test data to prevent leakage.
    Columns that are entirely NaN in the training set are filled with 0.
    """
    medians = np.nanmedian(X_train, axis=0)
    medians[np.isnan(medians)] = 0.0   # all-NaN column → fill with 0

    result = []
    for X in (X_train, X_val, X_test):
        X = X.copy()
        nan_rows, nan_cols = np.where(np.isnan(X))
        if nan_rows.size:
            X[nan_rows, nan_cols] = medians[nan_cols]
        result.append(X)

    return tuple(result)  # type: ignore[return-value]
