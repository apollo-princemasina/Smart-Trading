"""
Stability Analyzer
==================
Measures window-to-window consistency of model predictions and metrics.

A stable model should produce similar metric values across all walk-forward
windows.  High variance across windows signals poor generalization regardless
of the mean performance level.

Stability Score
---------------
Composite score in [0, 1] where 1 = perfectly stable.
Derived from the coefficient of variation (CV = std/|mean|) of key metrics.
Lower CV → higher stability.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metrics import aggregate_metric_stats

logger = logging.getLogger(__name__)

_NaN = float("nan")

# Metrics used for the composite stability score and their weights
_STABILITY_METRICS = {
    "f1":                   0.30,
    "accuracy":             0.20,
    "roc_auc":              0.20,
    "directional_accuracy": 0.15,
    "balanced_accuracy":    0.15,
}

_REG_STABILITY_METRICS = {
    "r2":   0.50,
    "mae":  0.25,
    "rmse": 0.25,
}


@dataclass
class StabilityResult:
    """Window-to-window stability analysis for a single model."""
    model_name:              str
    n_windows:               int
    metric_stats:            dict[str, dict]   # per-metric aggregate stats
    metric_cvs:              dict[str, float]  # coefficient of variation per metric
    confidence_cv:           Optional[float]   # CV of avg_confidence across windows
    stability_score:         float             # composite stability score [0, 1]
    most_variable_metric:    Optional[str]
    least_variable_metric:   Optional[str]
    window_scores:           list[float]       # per-window primary metric values
    prediction_variance:     float             # mean variance in prediction confidence
    is_stable:               bool              # True if stability_score >= threshold
    stability_threshold:     float


def analyze_stability(
    window_metric_dicts: list[dict],
    model_name:          str,
    task_type:           str = "classification",
    stability_threshold: float = 0.7,
) -> StabilityResult:
    """Compute stability metrics from per-window metric dictionaries.

    Args:
        window_metric_dicts: List of combined metric dicts, one per window.
                             Each dict contains classification + trading metrics.
        model_name:          Model name for labeling.
        task_type:           "classification" or "regression".
        stability_threshold: Minimum stability_score to be considered stable.

    Returns:
        StabilityResult with all stability measures.
    """
    n = len(window_metric_dicts)
    if n == 0:
        return StabilityResult(
            model_name=model_name, n_windows=0,
            metric_stats={}, metric_cvs={},
            confidence_cv=_NaN, stability_score=0.0,
            most_variable_metric=None, least_variable_metric=None,
            window_scores=[], prediction_variance=_NaN,
            is_stable=False, stability_threshold=stability_threshold,
        )

    # ── Collect per-metric value lists ─────────────────────────────────────────
    all_keys: set[str] = set()
    for d in window_metric_dicts:
        all_keys.update(k for k, v in d.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool))

    metric_values: dict[str, list[float]] = {k: [] for k in all_keys}
    for d in window_metric_dicts:
        for k in all_keys:
            v = d.get(k, _NaN)
            metric_values[k].append(float(v) if v == v else _NaN)

    # ── Aggregate stats per metric ─────────────────────────────────────────────
    metric_stats: dict[str, dict] = {}
    metric_cvs:   dict[str, float] = {}
    for k, vals in metric_values.items():
        stats = aggregate_metric_stats(vals)
        metric_stats[k] = stats
        metric_cvs[k]   = stats["cv"] if stats["cv"] == stats["cv"] else _NaN

    # ── Confidence stability ───────────────────────────────────────────────────
    confidence_vals = metric_values.get("avg_confidence", [])
    confidence_cv   = aggregate_metric_stats(confidence_vals).get("cv", _NaN)

    # ── Composite stability score ──────────────────────────────────────────────
    weights = _STABILITY_METRICS if task_type == "classification" else _REG_STABILITY_METRICS
    score_parts, weight_sum = [], 0.0
    for metric, w in weights.items():
        cv = metric_cvs.get(metric, _NaN)
        if cv == cv and cv >= 0:               # valid CV
            # Map CV → stability contribution: 0 CV = 1.0, high CV → 0
            contribution = 1.0 / (1.0 + cv)
            score_parts.append(w * contribution)
            weight_sum  += w
    stability_score = float(sum(score_parts) / weight_sum) if weight_sum > 0 else 0.0

    # ── Most / least variable metric ──────────────────────────────────────────
    valid_cvs = {k: v for k, v in metric_cvs.items() if v == v}
    most_var  = max(valid_cvs, key=valid_cvs.__getitem__) if valid_cvs else None
    least_var = min(valid_cvs, key=valid_cvs.__getitem__) if valid_cvs else None

    # ── Per-window primary metric values ──────────────────────────────────────
    primary = "f1" if task_type == "classification" else "r2"
    window_scores = [float(d.get(primary, _NaN)) for d in window_metric_dicts]

    # ── Prediction variance ────────────────────────────────────────────────────
    # Mean std of avg_confidence across windows (proxy for confidence instability)
    conf_vals_clean = [v for v in confidence_vals if v == v]
    pred_variance   = float(np.std(conf_vals_clean)) if len(conf_vals_clean) > 1 else _NaN

    return StabilityResult(
        model_name            = model_name,
        n_windows             = n,
        metric_stats          = metric_stats,
        metric_cvs            = metric_cvs,
        confidence_cv         = confidence_cv,
        stability_score       = stability_score,
        most_variable_metric  = most_var,
        least_variable_metric = least_var,
        window_scores         = window_scores,
        prediction_variance   = pred_variance,
        is_stable             = stability_score >= stability_threshold,
        stability_threshold   = stability_threshold,
    )


class StabilityAnalyzer:
    """Thin wrapper around :func:`analyze_stability` for object-oriented use."""

    def __init__(self, task_type: str = "classification", threshold: float = 0.7) -> None:
        self.task_type = task_type
        self.threshold = threshold

    def analyze(
        self,
        window_metric_dicts: list[dict],
        model_name:          str,
    ) -> StabilityResult:
        return analyze_stability(
            window_metric_dicts,
            model_name,
            self.task_type,
            self.threshold,
        )
