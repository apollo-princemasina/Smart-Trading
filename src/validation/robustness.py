"""
Robustness Analyzer
===================
Measures model robustness and generalization across walk-forward windows.

Robustness Analysis
-------------------
  Per-metric aggregate statistics: mean, median, min, max, std, CV
  Best/worst window identification
  Performance stability index

Generalization Analysis
-----------------------
  Overfitting detection:    train >> test by more than a threshold
  Underfitting detection:   both train and test are below minimum threshold
  Performance degradation:  linear trend in per-window scores is negative
  Regime sensitivity:       std of performance across high-vol vs low-vol windows

Market Regime Proxies
---------------------
Because raw price data is not available in this module, regimes are inferred
from the variance of the target variable across windows:
  High volatility  — target entropy > median
  Low volatility   — target entropy ≤ median
  Trending         — one class dominates (imbalanced target distribution)
  Ranging          — balanced target distribution
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metrics import aggregate_metric_stats

logger = logging.getLogger(__name__)
_NaN = float("nan")


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class GeneralizationAnalysis:
    """Overfitting, underfitting, degradation, and regime sensitivity."""
    overfitting_detected:      bool
    underfitting_detected:     bool
    performance_degradation:   bool
    regime_sensitivity:        float       # std of scores across regimes
    high_vol_performance:      Optional[float]
    low_vol_performance:       Optional[float]
    trending_performance:      Optional[float]
    ranging_performance:       Optional[float]
    train_test_gap:            Optional[float]  # train_metric - test_metric
    degradation_slope:         float           # linear trend coefficient
    generalization_score:      float           # composite [0, 1]


@dataclass
class RobustnessResult:
    """Robustness summary for a single model across all windows."""
    model_name:      str
    n_windows:       int
    # Per-metric aggregate statistics
    metric_stats:    dict[str, dict]   # {metric_name: {mean, median, min, max, std, cv}}
    # Window-level identification
    best_window:     Optional[int]     # window_number with highest primary metric
    worst_window:    Optional[int]     # window_number with lowest primary metric
    best_score:      float
    worst_score:     float
    # Composite scores
    robustness_score:     float        # composite [0, 1]
    generalization:       GeneralizationAnalysis
    # Primary metric name used for ranking
    primary_metric:       str


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_robustness(
    window_results:          list,           # list of WindowValidationResult
    model_name:              str,
    task_type:               str = "classification",
    train_metrics_history:   Optional[list[dict]] = None,
    overfitting_threshold:   float = 0.15,
    min_acceptable_primary:  float = 0.30,
    primary_metric:          Optional[str] = None,
) -> RobustnessResult:
    """Compute robustness and generalization metrics.

    Args:
        window_results:         List of WindowValidationResult objects.
        model_name:             Model identifier.
        task_type:              "classification" or "regression".
        train_metrics_history:  Optional list of training metric dicts per window.
                                If provided, used for overfitting detection.
        overfitting_threshold:  train - test gap above which overfitting is flagged.
        min_acceptable_primary: Minimum acceptable primary metric value.
        primary_metric:         Name of primary metric. Defaults to "f1" or "r2".

    Returns:
        RobustnessResult.
    """
    n = len(window_results)
    primary = primary_metric or ("f1" if task_type == "classification" else "r2")

    if n == 0:
        return RobustnessResult(
            model_name=model_name, n_windows=0, metric_stats={},
            best_window=None, worst_window=None, best_score=_NaN, worst_score=_NaN,
            robustness_score=0.0,
            generalization=_empty_generalization(),
            primary_metric=primary,
        )

    # ── Extract per-window metric dicts ───────────────────────────────────────
    def _get_metrics(r) -> dict:
        m = dict(r.combined_metrics or {})
        return m

    window_metrics = [_get_metrics(r) for r in window_results]
    window_numbers = [r.window_number for r in window_results]

    # ── Aggregate stats per metric ─────────────────────────────────────────────
    all_keys: set[str] = set()
    for d in window_metrics:
        all_keys.update(
            k for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )

    metric_stats: dict[str, dict] = {}
    for k in all_keys:
        vals = [float(d.get(k, _NaN)) for d in window_metrics]
        metric_stats[k] = aggregate_metric_stats(vals)

    # ── Best / worst window ────────────────────────────────────────────────────
    primary_vals = [float(d.get(primary, _NaN)) for d in window_metrics]
    valid_pairs  = [(v, w) for v, w in zip(primary_vals, window_numbers) if v == v]

    if valid_pairs:
        best_score,  best_window  = max(valid_pairs, key=lambda x: x[0])
        worst_score, worst_window = min(valid_pairs, key=lambda x: x[0])
    else:
        best_score = worst_score = _NaN
        best_window = worst_window = None

    # ── Composite robustness score ─────────────────────────────────────────────
    robustness_score = _compute_robustness_score(metric_stats, task_type)

    # ── Generalization analysis ────────────────────────────────────────────────
    gen = _analyze_generalization(
        window_results       = window_results,
        window_metrics       = window_metrics,
        primary_vals         = primary_vals,
        primary              = primary,
        task_type            = task_type,
        train_metrics_history = train_metrics_history,
        overfitting_threshold = overfitting_threshold,
        min_acceptable       = min_acceptable_primary,
    )

    return RobustnessResult(
        model_name       = model_name,
        n_windows        = n,
        metric_stats     = metric_stats,
        best_window      = best_window,
        worst_window     = worst_window,
        best_score       = best_score,
        worst_score      = worst_score,
        robustness_score = robustness_score,
        generalization   = gen,
        primary_metric   = primary,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_robustness_score(
    metric_stats: dict[str, dict],
    task_type:    str,
) -> float:
    """Compute a composite robustness score in [0, 1].

    A high score means good mean performance AND low variance across windows.
    Score = geometric mean of (1 - CV_norm) weighted by mean performance.
    """
    if task_type == "classification":
        key_metrics = ["f1", "accuracy", "roc_auc", "directional_accuracy"]
    else:
        key_metrics = ["r2", "mae"]

    contributions = []
    for k in key_metrics:
        stats = metric_stats.get(k)
        if stats is None:
            continue
        mean = stats["mean"]
        cv   = stats["cv"]
        if mean != mean or cv != cv:   # NaN
            continue
        # Normalize: mean_score × stability
        mean_component = max(0.0, float(mean))
        stab_component = 1.0 / (1.0 + max(0.0, float(cv)))
        contributions.append(mean_component * stab_component)

    if not contributions:
        return 0.0
    return float(np.mean(contributions))


def _analyze_generalization(
    window_results:       list,
    window_metrics:       list[dict],
    primary_vals:         list[float],
    primary:              str,
    task_type:            str,
    train_metrics_history: Optional[list[dict]],
    overfitting_threshold: float,
    min_acceptable:       float,
) -> GeneralizationAnalysis:
    n = len(window_results)

    # ── Overfitting detection ─────────────────────────────────────────────────
    train_test_gap   = _NaN
    overfitting      = False
    if train_metrics_history and len(train_metrics_history) == n:
        train_vals = [float(d.get(primary, _NaN)) for d in train_metrics_history]
        gaps       = [t - v for t, v in zip(train_vals, primary_vals)
                      if t == t and v == v]
        if gaps:
            train_test_gap = float(np.mean(gaps))
            overfitting    = train_test_gap > overfitting_threshold

    # ── Underfitting detection ────────────────────────────────────────────────
    valid_primary = [v for v in primary_vals if v == v]
    underfitting  = bool(valid_primary and np.mean(valid_primary) < min_acceptable)

    # ── Performance degradation (linear trend) ────────────────────────────────
    degradation_slope = _NaN
    degradation       = False
    valid_pairs = [(i, v) for i, v in enumerate(primary_vals) if v == v]
    if len(valid_pairs) >= 3:
        xs  = np.array([p[0] for p in valid_pairs], dtype=float)
        ys  = np.array([p[1] for p in valid_pairs], dtype=float)
        coeffs        = np.polyfit(xs, ys, 1)
        degradation_slope = float(coeffs[0])
        # Flag if slope is meaningfully negative relative to mean
        mean_score = float(np.mean(ys))
        if mean_score > 0 and degradation_slope < -(mean_score * 0.05):
            degradation = True

    # ── Regime sensitivity ─────────────────────────────────────────────────────
    # Infer regimes from target label entropy per window
    target_entropies = []
    for r in window_results:
        y = getattr(r, "y_true", None)
        if y is not None and len(y) > 0:
            target_entropies.append(_label_entropy(y))
        else:
            target_entropies.append(None)

    hi_vol_scores   = []
    lo_vol_scores   = []
    trend_scores    = []
    range_scores    = []
    entropy_values  = [e for e in target_entropies if e is not None]
    median_entropy  = float(np.median(entropy_values)) if entropy_values else 0.5

    for i, (entropy, pval) in enumerate(zip(target_entropies, primary_vals)):
        if pval != pval:   # NaN
            continue
        if entropy is None:
            continue
        if entropy > median_entropy:
            hi_vol_scores.append(pval)
        else:
            lo_vol_scores.append(pval)
        # Trending = low entropy (one class dominates)
        if entropy < 0.6:
            trend_scores.append(pval)
        else:
            range_scores.append(pval)

    regime_sensitivity = _NaN
    if hi_vol_scores and lo_vol_scores:
        regime_sensitivity = float(abs(np.mean(hi_vol_scores) - np.mean(lo_vol_scores)))

    # ── Generalization score ──────────────────────────────────────────────────
    gen_parts = []
    if not overfitting:
        gen_parts.append(1.0)
    else:
        gen_parts.append(max(0.0, 1.0 - min(train_test_gap, 1.0)))
    if not degradation:
        gen_parts.append(1.0)
    else:
        # Scale degradation penalty by slope magnitude
        gen_parts.append(max(0.0, 1.0 - min(abs(degradation_slope), 1.0)))
    if regime_sensitivity == regime_sensitivity:
        gen_parts.append(max(0.0, 1.0 - min(regime_sensitivity, 1.0)))
    gen_score = float(np.mean(gen_parts)) if gen_parts else 0.0

    return GeneralizationAnalysis(
        overfitting_detected    = overfitting,
        underfitting_detected   = underfitting,
        performance_degradation = degradation,
        regime_sensitivity      = regime_sensitivity,
        high_vol_performance    = float(np.mean(hi_vol_scores))  if hi_vol_scores  else _NaN,
        low_vol_performance     = float(np.mean(lo_vol_scores))  if lo_vol_scores  else _NaN,
        trending_performance    = float(np.mean(trend_scores))   if trend_scores   else _NaN,
        ranging_performance     = float(np.mean(range_scores))   if range_scores   else _NaN,
        train_test_gap          = train_test_gap,
        degradation_slope       = degradation_slope,
        generalization_score    = gen_score,
    )


def _label_entropy(y: np.ndarray) -> float:
    """Shannon entropy of a label array (normalized to [0, 1])."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) <= 1:
        return 0.0
    probs = counts / counts.sum()
    raw   = -float(np.sum(probs * np.log2(probs + 1e-12)))
    return raw / np.log2(len(classes))   # normalize by max entropy


def _empty_generalization() -> GeneralizationAnalysis:
    nan = _NaN
    return GeneralizationAnalysis(
        overfitting_detected=False, underfitting_detected=False,
        performance_degradation=False, regime_sensitivity=nan,
        high_vol_performance=nan, low_vol_performance=nan,
        trending_performance=nan, ranging_performance=nan,
        train_test_gap=nan, degradation_slope=nan, generalization_score=0.0,
    )


class RobustnessAnalyzer:
    """Object-oriented wrapper around :func:`analyze_robustness`."""

    def __init__(
        self,
        task_type:             str   = "classification",
        overfitting_threshold: float = 0.15,
        min_acceptable:        float = 0.30,
    ) -> None:
        self.task_type             = task_type
        self.overfitting_threshold = overfitting_threshold
        self.min_acceptable        = min_acceptable

    def analyze(
        self,
        window_results:        list,
        model_name:            str,
        train_metrics_history: Optional[list[dict]] = None,
    ) -> RobustnessResult:
        return analyze_robustness(
            window_results          = window_results,
            model_name              = model_name,
            task_type               = self.task_type,
            train_metrics_history   = train_metrics_history,
            overfitting_threshold   = self.overfitting_threshold,
            min_acceptable_primary  = self.min_acceptable,
        )
