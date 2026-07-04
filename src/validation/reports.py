"""
Validation Reports
==================
Generates six report artifacts from ValidationPipelineResult.

Output files
------------
  walk_forward_validation_report.md   — executive narrative summary
  validation_summary.csv              — one row per model
  window_metrics.csv                  — one row per model × window
  robustness_report.md                — robustness analysis narrative
  generalization_report.md            — overfitting/underfitting narrative
  stability_report.md                 — window-to-window stability narrative
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def generate_all_reports(
    pipeline_result: Any,   # ValidationPipelineResult
    output_dir:      Path,
    task_type:       str = "classification",
) -> dict[str, Path]:
    """Write all six report files.

    Returns:
        Dict mapping artifact name → Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["walk_forward_validation_report.md"] = _write_main_report(
        pipeline_result, output_dir, task_type
    )
    paths["validation_summary.csv"]  = _write_validation_summary(pipeline_result, output_dir)
    paths["window_metrics.csv"]      = _write_window_metrics(pipeline_result, output_dir, task_type)
    paths["robustness_report.md"]    = _write_robustness_report(pipeline_result, output_dir, task_type)
    paths["generalization_report.md"] = _write_generalization_report(pipeline_result, output_dir)
    paths["stability_report.md"]     = _write_stability_report(pipeline_result, output_dir)
    return paths


# ── Main report ───────────────────────────────────────────────────────────────

def _write_main_report(result: Any, out: Path, task_type: str) -> Path:
    summary = result.overall_summary or {}
    ranked  = result.ranked_models or []
    lines   = [
        "# Walk-Forward Validation Report",
        "",
        "## Executive Summary",
        "",
        f"- Task type:          {task_type}",
        f"- Windows evaluated:  {result.n_windows}",
        f"- Models evaluated:   {result.n_models}",
        f"- Total time:         {result.total_time_s:.1f}s",
        f"- Best model:         **{summary.get('best_model', 'N/A')}**",
        "",
    ]

    # Model ranking table
    if result.model_results:
        prim = "F1" if task_type == "classification" else "R²"
        lines += [
            "## Model Rankings",
            "",
            f"| Rank | Model | Status | Ranking Score | Stability | Robustness | {prim} Mean |",
            f"|---|---|---|---|---|---|---|",
        ]
        for rank, mvr in enumerate(result.model_results, 1):
            prim_stats = mvr.robustness.metric_stats.get(
                "f1" if task_type == "classification" else "r2", {}
            )
            prim_mean = prim_stats.get("mean")
            prim_str  = f"{prim_mean:.4f}" if prim_mean is not None and _is_valid(prim_mean) else "N/A"
            lines.append(
                f"| {rank} | {mvr.model_name} | {_status_badge(mvr.acceptance_status)} "
                f"| {mvr.ranking_score:.4f} | {mvr.stability.stability_score:.4f} "
                f"| {mvr.robustness.robustness_score:.4f} | {prim_str} |"
            )
        lines.append("")

    # Acceptance summary
    lines += ["## Acceptance Summary", ""]
    for mvr in result.model_results:
        icon = "✓" if mvr.acceptance_status == "production_ready" else (
            "⚠" if mvr.acceptance_status == "needs_improvement" else "✗"
        )
        lines.append(f"### {icon} {mvr.model_name} — {mvr.acceptance_status.replace('_', ' ').title()}")
        if mvr.acceptance_reasons:
            for r in mvr.acceptance_reasons:
                lines.append(f"  - {r}")
        else:
            lines.append("  - All thresholds passed")
        lines.append("")

    # Errors
    if result.errors:
        lines += ["## Errors", ""]
        for e in result.errors:
            lines.append(f"- {e}")
        lines.append("")

    path = out / "walk_forward_validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ── validation_summary.csv ────────────────────────────────────────────────────

def _write_validation_summary(result: Any, out: Path) -> Path:
    fieldnames = [
        "rank", "model_name", "acceptance_status", "ranking_score",
        "stability_score", "robustness_score", "generalization_score",
        "mean_inference_ms", "n_windows", "n_successful_windows",
        "primary_metric_mean", "primary_metric_std",
        "directional_accuracy_mean", "roc_auc_mean",
        "best_window", "worst_window",
        "overfitting_detected", "performance_degradation",
    ]
    rows = []
    for rank, mvr in enumerate(result.model_results, 1):
        pm_stats = mvr.robustness.metric_stats.get("f1" if mvr.task_type == "classification" else "r2", {})
        da_stats = mvr.robustness.metric_stats.get("directional_accuracy", {})
        au_stats = mvr.robustness.metric_stats.get("roc_auc", {})
        n_ok     = sum(1 for r in mvr.window_results if r.error is None)
        rows.append({
            "rank":                  rank,
            "model_name":            mvr.model_name,
            "acceptance_status":     mvr.acceptance_status,
            "ranking_score":         mvr.ranking_score,
            "stability_score":       mvr.stability.stability_score,
            "robustness_score":      mvr.robustness.robustness_score,
            "generalization_score":  mvr.robustness.generalization.generalization_score,
            "mean_inference_ms":     mvr.mean_inference_ms,
            "n_windows":             mvr.stability.n_windows,
            "n_successful_windows":  n_ok,
            "primary_metric_mean":   pm_stats.get("mean"),
            "primary_metric_std":    pm_stats.get("std"),
            "directional_accuracy_mean": da_stats.get("mean"),
            "roc_auc_mean":          au_stats.get("mean"),
            "best_window":           mvr.robustness.best_window,
            "worst_window":          mvr.robustness.worst_window,
            "overfitting_detected":  mvr.robustness.generalization.overfitting_detected,
            "performance_degradation": mvr.robustness.generalization.performance_degradation,
        })
    path = out / "validation_summary.csv"
    _write_csv(path, fieldnames, rows)
    return path


# ── window_metrics.csv ────────────────────────────────────────────────────────

def _write_window_metrics(result: Any, out: Path, task_type: str) -> Path:
    if task_type == "classification":
        metric_keys = [
            "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "roc_auc", "pr_auc", "mcc", "cohen_kappa",
            "directional_accuracy", "long_accuracy", "short_accuracy",
            "tp_prediction_accuracy", "sl_prediction_accuracy",
            "avg_confidence", "expected_return", "expected_risk",
            "risk_reward_accuracy",
        ]
    else:
        metric_keys = ["mae", "rmse", "mape", "r2"]

    fieldnames = ["window_number", "model_name", "n_test", "inference_time_s", "error"] + metric_keys
    rows = []
    for mvr in result.model_results:
        for r in mvr.window_results:
            row = {
                "window_number":   r.window_number,
                "model_name":      r.model_name,
                "n_test":          r.n_test,
                "inference_time_s": r.inference_time_s,
                "error":           r.error or "",
            }
            for k in metric_keys:
                row[k] = r.combined_metrics.get(k, "") if r.combined_metrics else ""
            rows.append(row)
    path = out / "window_metrics.csv"
    _write_csv(path, fieldnames, rows)
    return path


# ── robustness_report.md ──────────────────────────────────────────────────────

def _write_robustness_report(result: Any, out: Path, task_type: str) -> Path:
    prim = "f1" if task_type == "classification" else "r2"
    lines = [
        "# Robustness Report",
        "",
        "Aggregate statistics for each model across all walk-forward windows.",
        "",
    ]
    for mvr in result.model_results:
        rob = mvr.robustness
        lines += [
            f"## {mvr.model_name}",
            "",
            f"- Robustness score:   {rob.robustness_score:.4f}",
            f"- Best window:        {rob.best_window}  (score={rob.best_score:.4f})",
            f"- Worst window:       {rob.worst_window} (score={rob.worst_score:.4f})",
            "",
        ]

        # Primary metric stats
        pm = rob.metric_stats.get(prim, {})
        if pm:
            lines += [
                f"### {prim.upper()} Statistics",
                "",
                f"| Stat   | Value  |",
                f"|--------|--------|",
                f"| Mean   | {_fmt(pm.get('mean'))} |",
                f"| Median | {_fmt(pm.get('median'))} |",
                f"| Min    | {_fmt(pm.get('min'))} |",
                f"| Max    | {_fmt(pm.get('max'))} |",
                f"| Std    | {_fmt(pm.get('std'))} |",
                f"| CV     | {_fmt(pm.get('cv'))} |",
                "",
            ]

        # Additional key metrics summary
        extra_keys = (
            ["accuracy", "roc_auc", "directional_accuracy"]
            if task_type == "classification"
            else ["mae", "rmse"]
        )
        has_extra = any(k in rob.metric_stats for k in extra_keys)
        if has_extra:
            lines += ["### Additional Metrics (Mean ± Std)", ""]
            for k in extra_keys:
                stats = rob.metric_stats.get(k, {})
                mean  = stats.get("mean")
                std   = stats.get("std")
                if mean is not None:
                    lines.append(f"- {k}: {_fmt(mean)} ± {_fmt(std)}")
            lines.append("")

    path = out / "robustness_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ── generalization_report.md ──────────────────────────────────────────────────

def _write_generalization_report(result: Any, out: Path) -> Path:
    lines = [
        "# Generalization Report",
        "",
        "Evaluates overfitting, underfitting, performance degradation, and regime sensitivity.",
        "",
    ]
    for mvr in result.model_results:
        gen = mvr.robustness.generalization
        lines += [
            f"## {mvr.model_name}",
            "",
            f"| Dimension               | Value |",
            f"|-------------------------|-------|",
            f"| Generalization score    | {gen.generalization_score:.4f} |",
            f"| Overfitting detected    | {'Yes ⚠' if gen.overfitting_detected else 'No'} |",
            f"| Underfitting detected   | {'Yes ⚠' if gen.underfitting_detected else 'No'} |",
            f"| Performance degradation | {'Yes ⚠' if gen.performance_degradation else 'No'} |",
            f"| Regime sensitivity      | {_fmt(gen.regime_sensitivity)} |",
            f"| Train-test gap          | {_fmt(gen.train_test_gap)} |",
            f"| Degradation slope       | {_fmt(gen.degradation_slope)} |",
            "",
            "**Market Regime Performance**",
            "",
            f"| Regime   | Mean Score |",
            f"|----------|------------|",
            f"| High Vol | {_fmt(gen.high_vol_performance)} |",
            f"| Low Vol  | {_fmt(gen.low_vol_performance)} |",
            f"| Trending | {_fmt(gen.trending_performance)} |",
            f"| Ranging  | {_fmt(gen.ranging_performance)} |",
            "",
        ]

    path = out / "generalization_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ── stability_report.md ───────────────────────────────────────────────────────

def _write_stability_report(result: Any, out: Path) -> Path:
    lines = [
        "# Stability Report",
        "",
        "Measures window-to-window consistency of model performance.",
        "",
    ]
    for mvr in result.model_results:
        stab = mvr.stability
        lines += [
            f"## {mvr.model_name}",
            "",
            f"- Stability score:   {stab.stability_score:.4f}  "
            f"({'STABLE' if stab.is_stable else 'UNSTABLE'})",
            f"- Most variable:     {stab.most_variable_metric}",
            f"- Least variable:    {stab.least_variable_metric}",
            f"- Confidence CV:     {_fmt(stab.confidence_cv)}",
            f"- Prediction std:    {_fmt(stab.prediction_variance)}",
            "",
        ]

        # Top CVs table
        top_cvs = sorted(
            ((k, v) for k, v in stab.metric_cvs.items() if _is_valid(v)),
            key=lambda x: x[1], reverse=True
        )[:8]
        if top_cvs:
            lines += [
                "### Coefficient of Variation (top variable metrics)",
                "",
                "| Metric | CV |",
                "|--------|----|",
            ]
            for k, cv in top_cvs:
                lines.append(f"| {k} | {cv:.4f} |")
            lines.append("")

        # Per-window primary scores
        if stab.window_scores:
            lines += [
                "### Per-Window Scores (primary metric)",
                "",
                "| Window | Score |",
                "|--------|-------|",
            ]
            for i, s in enumerate(stab.window_scores, 1):
                lines.append(f"| {i:03d} | {_fmt(s)} |")
            lines.append("")

    path = out / "stability_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def _fmt(v: Optional[float]) -> str:
    if v is None or v != v:   # None or NaN
        return "N/A"
    return f"{v:.4f}"


def _is_valid(v) -> bool:
    return v is not None and v == v  # not None and not NaN


def _status_badge(status: str) -> str:
    badges = {
        "production_ready":  "✓ Ready",
        "needs_improvement": "⚠ Improve",
        "rejected":          "✗ Rejected",
    }
    return badges.get(status, status)
