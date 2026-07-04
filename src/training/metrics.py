"""
Training Metrics
================
Pure functions for computing classification, regression, and trading-specific
metrics.  No model or data dependencies — just numpy arrays in, dicts out.

Classification metrics
----------------------
accuracy, precision, recall, f1, roc_auc, pr_auc, log_loss, confusion_matrix

Regression metrics
------------------
mae, rmse, mse, mape, r2

Trading metrics (classification only)
--------------------------------------
directional_accuracy, tp_prediction_accuracy, sl_prediction_accuracy,
avg_confidence, prediction_distribution, per_class_precision
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

_NaN = float("nan")


# ── Task detection ────────────────────────────────────────────────────────────

def detect_task_type(y: pd.Series, n_unique_threshold: int = 20) -> str:
    """Return 'classification' or 'regression' based on target values.

    Heuristic: float dtype with more than *n_unique_threshold* distinct values
    is treated as a regression target; everything else as classification.
    """
    n_unique = int(y.dropna().nunique())
    if pd.api.types.is_float_dtype(y) and n_unique > n_unique_threshold:
        return "regression"
    return "classification"


# ── Classification ────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_prob:  Optional[np.ndarray],
    average: str = "macro",
) -> dict:
    """Compute standard classification metrics.

    Args:
        y_true:  Ground truth labels.
        y_pred:  Predicted labels.
        y_prob:  Predicted probabilities, shape (n, n_classes). None if unavailable.
        average: Averaging strategy for multi-class metrics.

    Returns:
        Dictionary with scalar metric values (lists/matrices for confusion_matrix).
    """
    classes   = np.unique(y_true)
    n_classes = len(classes)
    is_binary = n_classes == 2

    metrics: dict = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "n_classes": int(n_classes),
        "support":   int(len(y_true)),
    }

    # Probability-based metrics
    if y_prob is not None:
        try:
            if is_binary:
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
                metrics["pr_auc"]  = float(average_precision_score(y_true, prob_pos))
            else:
                # Ensure y_prob columns align with classes
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr",
                                  average="macro", labels=classes)
                )
                pr_aucs = []
                for i, cls in enumerate(classes):
                    y_bin = (y_true == cls).astype(int)
                    pr_aucs.append(float(average_precision_score(y_bin, y_prob[:, i])))
                metrics["pr_auc"] = float(np.mean(pr_aucs))

            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception as exc:
            logger.debug("Probability metric computation failed: %s", exc)
            metrics.setdefault("roc_auc", _NaN)
            metrics.setdefault("pr_auc",  _NaN)
            metrics.setdefault("log_loss", _NaN)
    else:
        metrics.update({"roc_auc": _NaN, "pr_auc": _NaN, "log_loss": _NaN})

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classes"]          = [int(c) if np.issubdtype(type(c), np.integer) else float(c)
                                   for c in classes]
    return metrics


# ── Regression ────────────────────────────────────────────────────────────────

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute standard regression metrics."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))

    nonzero = y_true != 0
    if nonzero.any():
        mape = float(np.mean(
            np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
        ) * 100)
    else:
        mape = _NaN

    return {
        "mae":     mae,
        "rmse":    rmse,
        "mse":     mse,
        "mape":    mape,
        "r2":      r2,
        "support": int(len(y_true)),
    }


# ── Trading ───────────────────────────────────────────────────────────────────

def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> dict:
    """Compute trading-specific evaluation metrics.

    Only meaningful for classification targets.  Assumes the label conventions
    used by the Label Generation module:
        class 0 → timeout / no trade
        class 1 → take profit (TP)
        class 2 → stop loss  (SL)

    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        y_prob: Predicted probabilities, shape (n, n_classes).

    Returns:
        dict with directional_accuracy, tp/sl prediction accuracy, confidence, distribution.
    """
    classes = np.unique(y_true)

    # Directional accuracy — fraction of correct class predictions
    directional_acc = float(accuracy_score(y_true, y_pred))

    # Per-class precision (precision when predicting that class)
    class_precision: dict[str, float] = {}
    for cls in classes:
        mask = y_pred == cls
        if mask.any():
            correct = ((y_pred == cls) & (y_true == cls)).sum()
            class_precision[str(int(cls))] = float(correct / mask.sum())
        else:
            class_precision[str(int(cls))] = _NaN

    # Average model confidence = mean of the max probability across samples
    if y_prob is not None and y_prob.ndim == 2:
        avg_confidence = float(np.max(y_prob, axis=1).mean())
    elif y_prob is not None:
        avg_confidence = float(np.mean(y_prob))
    else:
        avg_confidence = _NaN

    # Prediction class distribution
    pred_dist: dict[str, int] = {}
    unique_pred, counts = np.unique(y_pred, return_counts=True)
    for u, c in zip(unique_pred, counts):
        pred_dist[str(int(u))] = int(c)

    return {
        "directional_accuracy":    directional_acc,
        "tp_prediction_accuracy":  class_precision.get("1", _NaN),
        "sl_prediction_accuracy":  class_precision.get("2", _NaN),
        "avg_confidence":          avg_confidence,
        "prediction_distribution": pred_dist,
        "per_class_precision":     class_precision,
    }
