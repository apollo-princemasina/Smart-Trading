"""
Walk-Forward Reports
====================
Generates ``walk_forward_report.md`` under ``reports/walk_forward/``.

The report summarises the full set of generated windows: coverage,
split sizes, validation status, and configuration used.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .window_metadata import WindowMeta

logger = logging.getLogger(__name__)


def generate_walk_forward_report(
    windows:      list[WindowMeta],
    config_dict:  dict,
    output_dir:   Path,
    symbol:       str = "",
) -> Path:
    """Write ``walk_forward_report.md`` to *output_dir*.

    Args:
        windows:     Ordered list of WindowMeta objects.
        config_dict: Serialised WindowConfig fields.
        output_dir:  Directory to write the report (created if absent).
        symbol:      Ticker symbol (for heading).

    Returns:
        Path to the written report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "walk_forward_report.md"

    lines: list[str] = []
    _h = lines.append

    _h(f"# Walk-Forward Validation Report")
    if symbol:
        _h(f"**Symbol:** {symbol}  ")
    _h(f"**Windows generated:** {len(windows)}  ")
    _h(f"**Window type:** {config_dict.get('window_type', 'N/A')}  ")
    _h("")

    # ── Configuration ────────────────────────────────────────────────────────
    _h("## Configuration")
    _h("")
    _h("| Parameter | Value |")
    _h("|-----------|-------|")
    for k, v in config_dict.items():
        _h(f"| {k} | {v} |")
    _h("")

    if not windows:
        _h("*No windows were generated.*")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    # ── Coverage ─────────────────────────────────────────────────────────────
    _h("## Dataset Coverage")
    _h("")
    all_starts = [w.train.start for w in windows if w.train.start]
    all_ends   = [w.test.end    for w in windows if w.test.end]
    if all_starts and all_ends:
        _h(f"- **Full span:** {min(all_starts)} → {max(all_ends)}")
    _h(f"- **Total windows:** {len(windows)}")
    passed = sum(1 for w in windows if w.validation_passed)
    _h(f"- **Validation passed:** {passed}/{len(windows)}")
    _h("")

    # ── Window table ──────────────────────────────────────────────────────────
    _h("## Window Summary")
    _h("")
    _h("| # | Train Start | Train End | Val Start | Val End | Test Start | Test End | "
       "Train rows | Val rows | Test rows | Valid |")
    _h("|---|------------|-----------|-----------|---------|------------|----------|"
       "-----------|----------|-----------|-------|")
    for w in windows:
        status = "✓" if w.validation_passed else "✗"
        _h(
            f"| {w.window_number:03d} "
            f"| {w.train.start[:10]} | {w.train.end[:10]} "
            f"| {w.val.start[:10]}   | {w.val.end[:10]} "
            f"| {w.test.start[:10]}  | {w.test.end[:10]} "
            f"| {w.train.row_count:,} | {w.val.row_count:,} | {w.test.row_count:,} "
            f"| {status} |"
        )
    _h("")

    # ── Aggregate statistics ─────────────────────────────────────────────────
    _h("## Aggregate Statistics")
    _h("")
    for split_name, attr in [("Train", "train"), ("Validation", "val"), ("Test", "test")]:
        counts = [getattr(w, attr).row_count for w in windows]
        if counts:
            _h(f"### {split_name}")
            _h(f"- Min rows: {min(counts):,}")
            _h(f"- Max rows: {max(counts):,}")
            _h(f"- Mean rows: {sum(counts)/len(counts):,.0f}")
            _h("")

    # ── Validation issues ─────────────────────────────────────────────────────
    failed = [w for w in windows if not w.validation_passed]
    if failed:
        _h("## Validation Issues")
        _h("")
        for w in failed:
            _h(f"### Window {w.window_number:03d}")
            for issue in w.validation_issues:
                _h(f"- {issue}")
            _h("")

    # ── Artefacts ─────────────────────────────────────────────────────────────
    _h("## Artefact Paths")
    _h("")
    for w in windows:
        if w.artefact_paths:
            _h(f"**Window {w.window_number:03d}**")
            for k, v in w.artefact_paths.items():
                _h(f"- `{k}`: `{v}`")
            _h("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Walk-forward report saved → %s", report_path)
    return report_path
