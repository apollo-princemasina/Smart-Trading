"""
Validation Metrics
==================
Extended metric computation for walk-forward validation.

Extends ``src.training.metrics`` with:
  Classification: balanced_accuracy, MCC, Cohen's kappa
  Trading:        long_accuracy, short_accuracy, expected_return,
                  expected_risk, risk_reward_accuracy

All functions accept raw numpy arrays and return plain dicts.
Nothing is imported at module level from optional dependencies — each function
imports what it needs so the module is importable even in minimal environments.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.training.metrics import (
    compute_classification_metrics as _base_clf_metrics,
    compute_regression_metrics as _base_reg_metrics,
    compute_trading_metrics as _base_trading_metrics,
)

logger = logging.getLogger(__name__)
_NaN = float("nan")


# ── Classification ────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_prob:  Optional[np.ndarray] = None,
    average: str = "macro",
) -> dict:
    """Full classification metrics including balanced accuracy, MCC, kappa."""
    metrics = _base_clf_metrics(y_true, y_pred, y_prob, average)

    try:
        from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["mcc"]               = float(matthews_corrcoef(y_true, y_pred))
        metrics["cohen_kappa"]       = float(cohen_kappa_score(y_true, y_pred))
    except Exception as exc:
        logger.debug("Extended classification metrics failed: %s", exc)
        metrics.setdefault("balanced_accuracy", _NaN)
        metrics.setdefault("mcc",               _NaN)
        metrics.setdefault("cohen_kappa",        _NaN)

    return metrics


# ── Regression ────────────────────────────────────────────────────────────────

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Regression metrics (delegates to training module)."""
    return _base_reg_metrics(y_true, y_pred)


# ── Trading ───────────────────────────────────────────────────────────────────

def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """Extended trading metrics.

    Label convention (from Label Generation Engine):
        class 0 → timeout / no trade (neutral)
        class 1 → take profit (TP) / long
        class 2 → stop loss  (SL) / short

    Additional metrics vs training module:
        long_accuracy          — recall for class 1
        short_accuracy         — recall for class 2
        expected_return        — avg_confidence × directional_accuracy (proxy)
        expected_risk          — std of max class probability
        risk_reward_accuracy   — tp_accuracy / sl_accuracy (higher is better)
    """
    base = _base_trading_metrics(y_true, y_pred, y_prob)

    classes = np.unique(y_true)

    # ── Long accuracy (recall for class 1) ────────────────────────────────────
    long_acc = _NaN
    if 1 in classes:
        mask = (y_true == 1)
        if mask.any():
            long_acc = float((y_pred[mask] == 1).sum() / mask.sum())
    base["long_accuracy"] = long_acc

    # ── Short accuracy (recall for class 2) ───────────────────────────────────
    short_acc = _NaN
    if 2 in classes:
        mask = (y_true == 2)
        if mask.any():
            short_acc = float((y_pred[mask] == 2).sum() / mask.sum())
    base["short_accuracy"] = short_acc

    # ── Expected return & risk ─────────────────────────────────────────────────
    if y_prob is not None and y_prob.ndim == 2:
        max_probs = np.max(y_prob, axis=1)
        base["expected_return"] = float(np.mean(max_probs) * base["directional_accuracy"])
        base["expected_risk"]   = float(np.std(max_probs))
    else:
        base["expected_return"] = _NaN
        base["expected_risk"]   = _NaN

    # ── Risk-reward accuracy ───────────────────────────────────────────────────
    tp_acc = base.get("tp_prediction_accuracy", _NaN)
    sl_acc = base.get("sl_prediction_accuracy", _NaN)
    if (tp_acc is not None and sl_acc is not None
            and sl_acc == sl_acc and sl_acc > 0          # not NaN, not zero
            and tp_acc == tp_acc):
        base["risk_reward_accuracy"] = float(tp_acc / sl_acc)
    else:
        base["risk_reward_accuracy"] = _NaN

    return base


# ── Aggregate across windows ──────────────────────────────────────────────────

def aggregate_metric_stats(values: list[float]) -> dict:
    """Compute descriptive stats for a list of per-window metric values.

    Ignores NaN entries.

    Returns:
        dict with mean, median, min, max, std, cv (coefficient of variation).
    """
    arr = np.array([v for v in values if v == v], dtype=float)   # drop NaN
    if len(arr) == 0:
        nan = _NaN
        return {"mean": nan, "median": nan, "min": nan, "max": nan,
                "std": nan, "cv": nan, "count": 0}
    mean   = float(np.mean(arr))
    std    = float(np.std(arr))
    cv     = float(std / abs(mean)) if mean != 0 else _NaN
    return {
        "mean":   mean,
        "median": float(np.median(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "std":    std,
        "cv":     cv,
        "count":  int(len(arr)),
    }
