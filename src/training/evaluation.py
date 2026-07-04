"""
Evaluation
==========
Aggregates per-window model results into summary tables used for model
comparison and leaderboard ranking.

Outputs
-------
aggregate_metrics()  → long-form DataFrame (one row per model × window × split)
model_comparison()   → wide-form DataFrame (one row per model × window)
build_leaderboard()  → ranked DataFrame (one row per model, averaged across windows)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .trainer import ModelWindowResult

logger = logging.getLogger(__name__)

_NaN = float("nan")

# Leaderboard scoring weights
_W_F1       = 0.40
_W_ROC_AUC  = 0.35
_W_DIR_ACC  = 0.25


def _nanmean(values: list) -> float:
    valid = [x for x in values if isinstance(x, (int, float)) and not np.isnan(x)]
    return float(np.mean(valid)) if valid else _NaN


def _scalar_metrics(metrics: dict) -> dict:
    """Return only scalar (non-list, non-dict) entries from a metrics dict."""
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))}


class Evaluator:
    """Aggregates ModelWindowResult objects into report-ready DataFrames."""

    # ------------------------------------------------------------------
    def aggregate_metrics(self, results: list[ModelWindowResult]) -> pd.DataFrame:
        """Return a long-form table — one row per (model, window, split).

        Columns: model, window, split, task_type, n_samples, training_time_s,
                 plus all scalar metric names.
        """
        rows = []
        for r in results:
            n_map = {"train": r.n_train, "val": r.n_val, "test": r.n_test}
            splits = [
                ("train", r.train_metrics, r.training_time_seconds),
                ("val",   r.val_metrics,   _NaN),
                ("test",  r.test_metrics,  _NaN),
            ]
            for split_name, metrics, train_time in splits:
                row: dict = {
                    "model":         r.model_name,
                    "window":        r.window_number,
                    "split":         split_name,
                    "task_type":     r.task_type,
                    "n_samples":     n_map[split_name],
                    "training_time_s": train_time,
                }
                row.update(_scalar_metrics(metrics))
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def model_comparison(self, results: list[ModelWindowResult]) -> pd.DataFrame:
        """Return a wide table — one row per (model, window).

        Key columns: model, window, n_train, n_val, n_test, training_time_s,
                     prediction_time_s, then {train|val|test}_{metric} for key metrics.
        """
        _KEY_METRICS = [
            "accuracy", "f1", "roc_auc", "pr_auc", "log_loss",
            "directional_accuracy", "avg_confidence",
            "mae", "rmse", "r2",
        ]
        rows = []
        for r in results:
            row: dict = {
                "model":            r.model_name,
                "window":           r.window_number,
                "task_type":        r.task_type,
                "n_train":          r.n_train,
                "n_val":            r.n_val,
                "n_test":           r.n_test,
                "training_time_s":  r.training_time_seconds,
                "prediction_time_s":r.prediction_time_seconds,
                "n_features":       r.n_features,
            }
            for split, metrics in [
                ("train", r.train_metrics),
                ("val",   r.val_metrics),
                ("test",  r.test_metrics),
            ]:
                for key in _KEY_METRICS:
                    val = metrics.get(key, _NaN)
                    if isinstance(val, (int, float)):
                        row[f"{split}_{key}"] = float(val)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def build_leaderboard(self, results: list[ModelWindowResult]) -> pd.DataFrame:
        """Return a ranked summary — one row per model, averaged over windows.

        Ranking key: composite_score = 0.40×F1 + 0.35×ROC-AUC + 0.25×DirAcc
        (computed on the *validation* split).

        For pure regression tasks, uses R² instead of F1/ROC-AUC/DirAcc.
        """
        if not results:
            logger.warning("No results to build leaderboard from.")
            return pd.DataFrame()

        model_names = sorted(set(r.model_name for r in results))
        rows = []

        for model_name in model_names:
            mr = [r for r in results if r.model_name == model_name]
            task = mr[0].task_type if mr else "classification"

            f1s       = [r.val_metrics.get("f1",       _NaN) for r in mr]
            roc_aucs  = [r.val_metrics.get("roc_auc",  _NaN) for r in mr]
            dir_accs  = [r.val_metrics.get("directional_accuracy",
                         r.val_metrics.get("accuracy", _NaN)) for r in mr]
            r2s       = [r.val_metrics.get("r2",       _NaN) for r in mr]
            maes      = [r.val_metrics.get("mae",      _NaN) for r in mr]
            test_f1s  = [r.test_metrics.get("f1",      _NaN) for r in mr]
            test_acc  = [r.test_metrics.get("accuracy",_NaN) for r in mr]
            times     = [r.training_time_seconds for r in mr]
            pred_ts   = [r.prediction_time_seconds for r in mr]

            mean_f1      = _nanmean(f1s)
            mean_roc     = _nanmean(roc_aucs)
            mean_dir     = _nanmean(dir_accs)
            mean_r2      = _nanmean(r2s)
            mean_mae     = _nanmean(maes)
            mean_test_f1 = _nanmean(test_f1s)
            mean_test_acc= _nanmean(test_acc)
            mean_time    = _nanmean(times)
            mean_pred    = _nanmean(pred_ts)

            # Composite score
            if task == "regression":
                composite = float(mean_r2) if not np.isnan(mean_r2) else 0.0
            else:
                composite = (
                    _W_F1      * (mean_f1  if not np.isnan(mean_f1)  else 0.0) +
                    _W_ROC_AUC * (mean_roc if not np.isnan(mean_roc) else 0.0) +
                    _W_DIR_ACC * (mean_dir if not np.isnan(mean_dir) else 0.0)
                )

            rows.append({
                "model":                    model_name,
                "n_windows":                len(mr),
                "task_type":                task,
                "mean_val_f1":              mean_f1,
                "mean_val_roc_auc":         mean_roc,
                "mean_val_directional_acc": mean_dir,
                "mean_val_r2":              mean_r2,
                "mean_val_mae":             mean_mae,
                "mean_test_f1":             mean_test_f1,
                "mean_test_accuracy":       mean_test_acc,
                "mean_training_time_s":     mean_time,
                "mean_prediction_time_s":   mean_pred,
                "composite_score":          composite,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df
