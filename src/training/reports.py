"""
Training Reports
================
Generates all four report artefacts from a completed training run:

1. training_report.md     — Human-readable Markdown summary.
2. model_comparison.csv   — Wide table: one row per (model × window).
3. metrics.csv            — Long table: one row per (model × window × split).
4. leaderboard.csv        — Ranked summary: one row per model.

All files are written to *output_dir* (typically ``reports/training/``).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import Evaluator
from .trainer import ModelWindowResult

logger = logging.getLogger(__name__)

_NaN = float("nan")


# ── Public entry points ───────────────────────────────────────────────────────

def generate_training_report(
    results:     list[ModelWindowResult],
    leaderboard: pd.DataFrame,
    config_dict: dict,
    output_dir:  Path,
    symbol:      str = "",
) -> Path:
    """Write ``training_report.md`` and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_report.md"

    lines: list[str] = []
    a = lines.append

    a("# Baseline Model Training Report")
    if symbol:
        a(f"**Symbol:** {symbol}  ")
    n_windows = len(set(r.window_number for r in results)) if results else 0
    n_models  = len(set(r.model_name   for r in results)) if results else 0
    a(f"**Windows trained:** {n_windows}  ")
    a(f"**Models per window:** {n_models}  ")
    a(f"**Total runs:** {len(results)}  ")
    a("")

    # ── Configuration ────────────────────────────────────────────────────────
    if config_dict:
        a("## Configuration")
        a("")
        a("| Parameter | Value |")
        a("|-----------|-------|")
        for k, v in config_dict.items():
            a(f"| {k} | {v} |")
        a("")

    if not results:
        a("*No training results available.*")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ── Leaderboard ──────────────────────────────────────────────────────────
    a("## Model Leaderboard (Validation)")
    a("")
    if not leaderboard.empty:
        a(_df_to_md(leaderboard[[
            "rank", "model", "mean_val_f1", "mean_val_roc_auc",
            "mean_val_directional_acc", "mean_val_r2",
            "mean_training_time_s", "composite_score",
        ]].round(4)))
    a("")

    # ── Per-window results ───────────────────────────────────────────────────
    a("## Results by Window")
    a("")
    windows = sorted(set(r.window_number for r in results))
    for wn in windows:
        wr = [r for r in results if r.window_number == wn]
        a(f"### Window {wn:03d}")
        a("")
        rows = []
        for r in wr:
            rows.append({
                "model":        r.model_name,
                "val_f1":       r.val_metrics.get("f1",       _NaN),
                "val_roc_auc":  r.val_metrics.get("roc_auc",  _NaN),
                "val_dir_acc":  r.val_metrics.get("directional_accuracy",
                                r.val_metrics.get("accuracy", _NaN)),
                "test_f1":      r.test_metrics.get("f1",      _NaN),
                "test_roc_auc": r.test_metrics.get("roc_auc", _NaN),
                "train_s":      round(r.training_time_seconds, 3),
            })
        a(_df_to_md(pd.DataFrame(rows).round(4)))
        a("")

    # ── Aggregate statistics ─────────────────────────────────────────────────
    a("## Aggregate Statistics (Validation F1)")
    a("")
    model_names = sorted(set(r.model_name for r in results))
    for mn in model_names:
        vals = [r.val_metrics.get("f1", _NaN) for r in results if r.model_name == mn]
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            a(f"- **{mn}**: mean={np.mean(valid):.4f}  "
              f"std={np.std(valid):.4f}  "
              f"min={np.min(valid):.4f}  "
              f"max={np.max(valid):.4f}")
    a("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Training report saved → %s", path)
    return path


def generate_metrics_csv(
    results:    list[ModelWindowResult],
    output_dir: Path,
) -> Path:
    """Write ``metrics.csv`` (long-form, one row per model × window × split)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.csv"

    ev = Evaluator()
    df = ev.aggregate_metrics(results)
    df.to_csv(path, index=False)
    logger.info("Metrics CSV saved → %s", path)
    return path


def generate_comparison_csv(
    results:    list[ModelWindowResult],
    output_dir: Path,
) -> Path:
    """Write ``model_comparison.csv`` (wide-form, one row per model × window)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_comparison.csv"

    ev = Evaluator()
    df = ev.model_comparison(results)
    df.to_csv(path, index=False)
    logger.info("Model comparison CSV saved → %s", path)
    return path


def generate_leaderboard_csv(
    leaderboard: pd.DataFrame,
    output_dir:  Path,
) -> Path:
    """Write ``leaderboard.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(path, index=False)
    logger.info("Leaderboard CSV saved → %s", path)
    return path


# ── Helper ────────────────────────────────────────────────────────────────────

def _df_to_md(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a GitHub-flavoured Markdown table."""
    if df.empty:
        return "*No data.*"

    def _fmt(v) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            return f"{v:.4f}" if abs(v) < 10_000 else f"{v:.2f}"
        return str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep    = "|" + "|".join([" --- " for _ in df.columns]) + "|"
    body   = "\n".join(
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for _, row in df.iterrows()
    )
    return "\n".join([header, sep, body])
