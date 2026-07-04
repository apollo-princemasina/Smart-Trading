"""
Model Selector
==============
Chooses the best model across all walk-forward windows and saves a canonical
``models/best_model/`` inference bundle.

Composite score (same weights as the training leaderboard):
    0.40 × F1  +  0.35 × ROC-AUC  +  0.25 × Directional-Accuracy   (classification)
    R²                                                                 (regression)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_COMPOSITE_WEIGHTS = {"f1": 0.40, "roc_auc": 0.35, "directional_accuracy": 0.25}


@dataclass
class SelectionResult:
    """Everything the caller needs to understand which bundle was chosen."""
    chosen_model_name:    str
    chosen_window_number: int
    composite_score:      float
    optimization_metric:  str
    val_score:            float
    baseline_val_score:   Optional[float]
    improvement_pct:      Optional[float]
    bundle_dir:           Path
    best_model_dir:       Path


class ModelSelector:
    """Select the best model/window combination and create the best bundle."""

    def select_best(
        self,
        results: list,           # list of WindowOptResult (from optimization_pipeline)
        task_type: str = "classification",
    ) -> Optional["WindowOptResult"]:
        """Return the WindowOptResult with the highest composite validation score.

        Args:
            results:   Non-empty list of WindowOptResult objects.
            task_type: "classification" or "regression".

        Returns:
            Best WindowOptResult, or None if *results* is empty.
        """
        if not results:
            return None

        scored = [(r, self._composite(r, task_type)) for r in results]
        best_result, best_score = max(scored, key=lambda x: x[1])
        logger.info(
            "Best model: %s (window %d) | composite=%.4f",
            best_result.model_name, best_result.window_number, best_score,
        )
        return best_result

    def compare_with_baseline(
        self,
        results: list,
        baseline_scores: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        """Build a comparison table of optimized vs baseline scores.

        Args:
            results:         List of WindowOptResult.
            baseline_scores: Dict mapping model_name → baseline_val_score.
                             Pass None to skip improvement calculation.

        Returns:
            List of comparison dicts, one per result.
        """
        rows = []
        for r in results:
            baseline = (baseline_scores or {}).get(r.model_name)
            imp = None
            if baseline is not None and baseline > 0:
                imp = round((r.best_val_score - baseline) / abs(baseline) * 100, 2)
            rows.append({
                "model_name":       r.model_name,
                "window_number":    r.window_number,
                "best_val_score":   r.best_val_score,
                "baseline_val_score": baseline,
                "improvement_pct":  imp,
                "n_trials":         r.n_trials_completed,
                "optimization_time_s": r.optimization_time_s,
            })
        return rows

    def create_best_bundle(
        self,
        best_result: "WindowOptResult",
        best_model_dir: Path,
    ) -> SelectionResult:
        """Copy the best window's bundle to *best_model_dir*.

        Args:
            best_result:    The WindowOptResult chosen by select_best().
            best_model_dir: Target path for the canonical best-model bundle.

        Returns:
            SelectionResult with all provenance fields.
        """
        from .artifact_manager import ArtifactManager

        source_dir = Path(best_result.bundle_dir)
        best_dir   = ArtifactManager.copy_to_best(source_dir, best_model_dir)

        task_type  = best_result.task_type
        val_metrics = best_result.val_metrics or {}
        composite  = self._composite_from_metrics(val_metrics, task_type)

        improvement = None
        if best_result.baseline_val_score is not None and best_result.baseline_val_score > 0:
            improvement = round(
                (best_result.best_val_score - best_result.baseline_val_score)
                / abs(best_result.baseline_val_score) * 100, 2,
            )

        return SelectionResult(
            chosen_model_name    = best_result.model_name,
            chosen_window_number = best_result.window_number,
            composite_score      = composite,
            optimization_metric  = best_result.optimization_metric,
            val_score            = best_result.best_val_score,
            baseline_val_score   = best_result.baseline_val_score,
            improvement_pct      = improvement,
            bundle_dir           = source_dir,
            best_model_dir       = best_dir,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _composite(result: Any, task_type: str) -> float:
        return ModelSelector._composite_from_metrics(
            result.val_metrics or {}, task_type
        )

    @staticmethod
    def _composite_from_metrics(val_metrics: dict, task_type: str) -> float:
        if task_type == "regression":
            return float(val_metrics.get("r2", 0.0))
        f1  = float(val_metrics.get("f1", 0.0) or 0.0)
        auc = float(val_metrics.get("roc_auc", 0.0) or 0.0)
        da  = float(val_metrics.get("directional_accuracy", 0.0) or 0.0)
        return (
            _COMPOSITE_WEIGHTS["f1"]                   * f1
            + _COMPOSITE_WEIGHTS["roc_auc"]            * auc
            + _COMPOSITE_WEIGHTS["directional_accuracy"] * da
        )
