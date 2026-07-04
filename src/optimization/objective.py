"""
Objective Function
==================
Defines the callable passed to ``study.optimize()``.

Each invocation:
  1. Suggests hyperparameters via the model's SearchSpace.
  2. Builds and fits the model on pre-computed training arrays.
  3. Evaluates the chosen metric on the validation set.
  4. Returns the scalar score (higher = better for all supported metrics).

Supported metrics
-----------------
f1                 — macro F1 score (classification)
roc_auc            — macro ROC-AUC (classification)
pr_auc             — macro PR-AUC (classification)
accuracy           — accuracy (classification)
directional_accuracy — same as accuracy; alias for trading context
tp_accuracy        — precision for class 1 (TP) (classification)
r2                 — coefficient of determination (regression)
neg_mae            — negative MAE, so higher = better (regression)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import optuna

from .search_space import SEARCH_SPACES, get_search_space

logger = logging.getLogger(__name__)

SUPPORTED_METRICS: list[str] = [
    "f1",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "directional_accuracy",
    "tp_accuracy",
    "r2",
    "neg_mae",
]

_CLASSIFICATION_METRICS = frozenset({
    "f1", "roc_auc", "pr_auc", "accuracy", "directional_accuracy", "tp_accuracy"
})
_REGRESSION_METRICS = frozenset({"r2", "neg_mae"})


# ── Score computation ─────────────────────────────────────────────────────────

def compute_objective_score(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    y_prob:   Optional[np.ndarray],
    metric:   str,
    task_type: str = "classification",
) -> float:
    """Compute a scalar score for a given metric.

    All classification metrics return values in [0, 1] (higher = better).
    Regression metrics may be negative.

    Raises:
        ValueError: If metric is not in SUPPORTED_METRICS.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from: {SUPPORTED_METRICS}."
        )

    if task_type == "regression":
        return _regression_score(y_true, y_pred, metric)

    return _classification_score(y_true, y_pred, y_prob, metric)


def _classification_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    metric: str,
) -> float:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    classes = np.unique(y_true)
    is_binary = len(classes) == 2

    try:
        if metric in ("accuracy", "directional_accuracy"):
            return float(accuracy_score(y_true, y_pred))

        if metric == "f1":
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        if metric == "tp_accuracy":
            mask = y_pred == 1
            if not mask.any():
                return 0.0
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            return float(tp / mask.sum())

        if metric in ("roc_auc", "pr_auc") and y_prob is None:
            return 0.0

        if metric == "roc_auc":
            if is_binary:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                return float(roc_auc_score(y_true, prob))
            return float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=classes
            ))

        if metric == "pr_auc":
            if is_binary:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                return float(average_precision_score(y_true, prob))
            aucs = [
                float(average_precision_score((y_true == c).astype(int), y_prob[:, i]))
                for i, c in enumerate(classes)
            ]
            return float(np.mean(aucs))

    except Exception as exc:
        logger.debug("Metric %s failed: %s", metric, exc)
        return 0.0

    return 0.0


def _regression_score(
    y_true: np.ndarray, y_pred: np.ndarray, metric: str
) -> float:
    from sklearn.metrics import mean_absolute_error, r2_score
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "neg_mae":
        return -float(mean_absolute_error(y_true, y_pred))
    return 0.0


# ── Objective callable ────────────────────────────────────────────────────────

class ObjectiveFunction:
    """Optuna-compatible objective.

    Pre-computed training and validation arrays are captured at construction
    time so each trial only does model creation, fitting, and scoring —
    no data extraction overhead.

    Args:
        model_name:  One of the supported model names.
        task_type:   "classification" or "regression".
        X_train:     Training features (imputed if sklearn model).
        y_train:     Training labels.
        X_val:       Validation features (same preprocessing as X_train).
        y_val:       Validation labels.
        metric:      Objective metric name from SUPPORTED_METRICS.
        random_seed: Passed to the model constructor.
        n_jobs:      Parallelism inside each model fit.
    """

    def __init__(
        self,
        model_name:  str,
        task_type:   str,
        X_train:     np.ndarray,
        y_train:     np.ndarray,
        X_val:       np.ndarray,
        y_val:       np.ndarray,
        metric:      str = "f1",
        random_seed: int = 42,
        n_jobs:      int = -1,
    ) -> None:
        self.model_name  = model_name
        self.task_type   = task_type
        self.X_train     = X_train
        self.y_train     = y_train
        self.X_val       = X_val
        self.y_val       = y_val
        self.metric      = metric
        self.random_seed = random_seed
        self.n_jobs      = n_jobs
        self._space      = get_search_space(model_name)

    def __call__(self, trial: optuna.Trial) -> float:
        params = self._space.suggest(trial)
        model  = self._space.build(params, self.task_type, self.random_seed, self.n_jobs)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_val)
        y_prob = model.predict_proba(self.X_val) if hasattr(model, "predict_proba") else None

        score = compute_objective_score(
            self.y_val, y_pred, y_prob, self.metric, self.task_type
        )
        logger.debug(
            "Trial %d | %s | %s=%.4f | params=%s",
            trial.number, self.model_name, self.metric, score,
            {k: v for k, v in params.items() if not k.startswith("rf_") and not k.startswith("et_")},
        )
        return score
