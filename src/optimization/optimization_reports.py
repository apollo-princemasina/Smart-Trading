"""
Optimization Reports
====================
Generates human-readable and machine-readable reports from the pipeline output.

Outputs:
  optimization_report.md     — narrative summary with tables
  optuna_results.csv         — one row per trial per model per window
  best_parameters.json       — best params for every model
  optimization_history.csv   — trial-by-trial value progression
  model_comparison.csv       — optimised vs baseline per model per window
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def generate_optimization_report(
    pipeline_result:  Any,          # PipelineOptResult
    output_dir:       Path,
    task_type:        str = "classification",
) -> dict[str, Path]:
    """Write all four report artifacts.

    Returns:
        Dict mapping artifact name → Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["optimization_report.md"]   = _write_markdown(pipeline_result, output_dir, task_type)
    paths["optuna_results.csv"]        = _write_optuna_results(pipeline_result, output_dir)
    paths["best_parameters.json"]      = _write_best_parameters(pipeline_result, output_dir)
    paths["optimization_history.csv"]  = _write_history(pipeline_result, output_dir)
    paths["model_comparison.csv"]      = _write_model_comparison(pipeline_result, output_dir)
    return paths


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_markdown(pipeline_result: Any, output_dir: Path, task_type: str) -> Path:
    lines = [
        "# Hyperparameter Optimization Report",
        "",
        "## Summary",
        "",
    ]

    sel = getattr(pipeline_result, "selection_result", None)
    if sel is not None:
        lines += [
            f"| Field | Value |",
            f"|---|---|",
            f"| Best model | `{sel.chosen_model_name}` |",
            f"| Best window | {sel.chosen_window_number} |",
            f"| Composite score | {sel.composite_score:.4f} |",
            f"| Optimization metric | {sel.optimization_metric} |",
            f"| Val score ({sel.optimization_metric}) | {sel.val_score:.4f} |",
        ]
        if sel.baseline_val_score is not None:
            lines.append(f"| Baseline val score | {sel.baseline_val_score:.4f} |")
        if sel.improvement_pct is not None:
            lines.append(f"| Improvement | {sel.improvement_pct:+.2f}% |")
        lines.append("")

    # per-window table
    results = getattr(pipeline_result, "results", [])
    if results:
        lines += [
            "## Per-Window Results",
            "",
            "| Window | Model | Trials | Best Val Score | Opt. Time (s) | Improved? |",
            "|---|---|---|---|---|---|",
        ]
        for r in results:
            imp = ""
            if getattr(r, "baseline_val_score", None) is not None and r.baseline_val_score > 0:
                delta = r.best_val_score - r.baseline_val_score
                imp   = f"{delta:+.4f}"
            lines.append(
                f"| {r.window_number:03d} "
                f"| {r.model_name} "
                f"| {r.n_trials_completed} "
                f"| {r.best_val_score:.4f} "
                f"| {r.optimization_time_s:.1f} "
                f"| {imp} |"
            )
        lines.append("")

    # aggregate stats
    if results:
        times  = [r.optimization_time_s for r in results]
        scores = [r.best_val_score for r in results]
        lines += [
            "## Aggregate Statistics",
            "",
            f"- Windows optimized:   {len(results)}",
            f"- Total trials:        {sum(r.n_trials_completed for r in results)}",
            f"- Total opt. time:     {sum(times):.1f}s",
            f"- Mean val score:      {sum(scores)/len(scores):.4f}",
            f"- Min  val score:      {min(scores):.4f}",
            f"- Max  val score:      {max(scores):.4f}",
            "",
        ]

    # errors
    errors = getattr(pipeline_result, "errors", [])
    if errors:
        lines += ["## Errors", ""]
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    path = output_dir / "optimization_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ── CSV artifacts ─────────────────────────────────────────────────────────────

def _write_optuna_results(pipeline_result: Any, output_dir: Path) -> Path:
    results = getattr(pipeline_result, "results", [])
    fieldnames = [
        "window_number", "model_name", "trial_number",
        "trial_value", "trial_state", "optimization_metric",
    ]
    rows = []
    for r in results:
        for t in getattr(r, "trial_history", []):
            rows.append({
                "window_number":       r.window_number,
                "model_name":          r.model_name,
                "trial_number":        t.get("number"),
                "trial_value":         t.get("value"),
                "trial_state":         t.get("state"),
                "optimization_metric": r.optimization_metric,
            })
    path = output_dir / "optuna_results.csv"
    _write_csv(path, fieldnames, rows)
    return path


def _write_best_parameters(pipeline_result: Any, output_dir: Path) -> Path:
    results = getattr(pipeline_result, "results", [])
    data = {}
    for r in results:
        key = f"{r.model_name}_w{r.window_number:03d}"
        data[key] = {
            "model_name":    r.model_name,
            "window_number": r.window_number,
            "best_params":   r.best_params,
            "best_val_score": r.best_val_score,
            "n_trials":      r.n_trials_completed,
        }
    path = output_dir / "best_parameters.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _write_history(pipeline_result: Any, output_dir: Path) -> Path:
    results = getattr(pipeline_result, "results", [])
    fieldnames = [
        "window_number", "model_name",
        "trial_number", "trial_value", "best_so_far",
    ]
    rows = []
    for r in results:
        best_so_far: Optional[float] = None
        for t in getattr(r, "trial_history", []):
            val = t.get("value")
            if val is not None:
                if best_so_far is None or val > best_so_far:
                    best_so_far = val
            rows.append({
                "window_number":  r.window_number,
                "model_name":     r.model_name,
                "trial_number":   t.get("number"),
                "trial_value":    val,
                "best_so_far":    best_so_far,
            })
    path = output_dir / "optimization_history.csv"
    _write_csv(path, fieldnames, rows)
    return path


def _write_model_comparison(pipeline_result: Any, output_dir: Path) -> Path:
    results = getattr(pipeline_result, "results", [])
    fieldnames = [
        "window_number", "model_name",
        "optimized_val_score", "baseline_val_score", "improvement_pct",
        "n_trials", "optimization_time_s",
    ]
    rows = []
    for r in results:
        baseline = getattr(r, "baseline_val_score", None)
        imp      = None
        if baseline is not None and baseline > 0:
            imp = round((r.best_val_score - baseline) / abs(baseline) * 100, 2)
        rows.append({
            "window_number":       r.window_number,
            "model_name":          r.model_name,
            "optimized_val_score": r.best_val_score,
            "baseline_val_score":  baseline,
            "improvement_pct":     imp,
            "n_trials":            r.n_trials_completed,
            "optimization_time_s": r.optimization_time_s,
        })
    path = output_dir / "model_comparison.csv"
    _write_csv(path, fieldnames, rows)
    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))
