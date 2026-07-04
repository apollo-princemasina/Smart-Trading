"""
Dataset Reports
===============
Generates human-readable artefacts for the assembled training dataset:

  reports/dataset/
    training_dataset_report.md     — comprehensive Markdown report
    training_dataset_metadata.json — serialised DatasetMeta

Generated files describe everything a data scientist or ML engineer needs
to understand the dataset before training a model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .dataset_metadata import DatasetMeta

logger = logging.getLogger(__name__)

_DEFAULT_REPORT_DIR = Path("reports") / "dataset"


class DatasetReportGenerator:
    """Generate all dataset report artefacts."""

    def __init__(self, report_dir: Optional[Path] = None) -> None:
        self.report_dir = Path(report_dir or _DEFAULT_REPORT_DIR)

    # ------------------------------------------------------------------
    def generate_all(
        self,
        dataset:  pd.DataFrame,
        metadata: DatasetMeta,
    ) -> dict[str, Path]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        paths["metadata"] = self._write_metadata(metadata)
        paths["report"]   = self._write_markdown(dataset, metadata)

        logger.info("DatasetReports: wrote %d artefacts → %s", len(paths), self.report_dir)
        return paths

    # ------------------------------------------------------------------
    def _write_metadata(self, meta: DatasetMeta) -> Path:
        path = self.report_dir / "training_dataset_metadata.json"
        meta.to_json(path)
        return path

    # ------------------------------------------------------------------
    def _write_markdown(self, dataset: pd.DataFrame, meta: DatasetMeta) -> Path:
        path = self.report_dir / "training_dataset_report.md"
        lines: list[str] = []

        # Header
        lines += [
            "# Training Dataset Report",
            "",
            f"**Dataset**: {meta.dataset_name} v{meta.dataset_version}  ",
            f"**Symbol**: {meta.symbol}  ",
            f"**Timeframe**: {meta.prediction_timeframe or 'n/a'}  ",
            f"**Generated**: {meta.created_at}  ",
            f"**Pipeline**: {meta.pipeline_version}  ",
            "",
            "---",
            "",
        ]

        # Dataset summary
        lines += [
            "## Dataset Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total rows | {meta.row_count:,} |",
            f"| Feature columns | {meta.feature_count} |",
            f"| Label columns | {meta.label_count} |",
            f"| Total columns | {meta.column_count} |",
            f"| Start date | {meta.start_date} |",
            f"| End date | {meta.end_date} |",
            f"| Missing values | {meta.missing_rate:.2%} |",
            f"| Feature set | {meta.feature_set} |",
            f"| Label version | v{meta.label_version} |",
            f"| Schema version | {meta.schema_version} |",
            f"| Validation | {'✓ PASSED' if meta.validation_passed else '✗ FAILED'} |",
            "",
            "---",
            "",
        ]

        # Feature summary
        feat_cols = [c for c in meta.column_summaries if not c.is_label]
        lines += ["## Feature Summary", ""]

        # Group features by prefix
        prefix_counts: dict[str, int] = {}
        for cs in feat_cols:
            parts = cs.name.split("_")
            pfx   = parts[0] if len(parts) > 1 else "other"
            prefix_counts[pfx] = prefix_counts.get(pfx, 0) + 1

        if prefix_counts:
            lines += ["| Prefix | Feature Count |", "|---|---|"]
            for pfx, cnt in sorted(prefix_counts.items(), key=lambda x: -x[1]):
                lines.append(f"| `{pfx}_` | {cnt} |")
            lines.append("")

        # Top features by NaN rate (cleanest first)
        top_feats = sorted(feat_cols, key=lambda c: c.nan_rate)[:20]
        if top_feats:
            lines += [
                "### Cleanest Features (by NaN rate)",
                "",
                "| Feature | Dtype | NaN Rate | Valid Rows |",
                "|---|---|---|---|",
            ]
            for cs in top_feats:
                lines.append(
                    f"| `{cs.name}` | {cs.dtype} | {cs.nan_rate:.1%} | {cs.n_valid:,} |"
                )
            lines.append("")

        lines += ["---", ""]

        # Label summary
        lbl_cols = [c for c in meta.column_summaries if c.is_label]
        lines += ["## Label Summary", ""]
        if lbl_cols:
            lines += [
                "| Label Column | Dtype | NaN Rate | Valid Rows | Value Range |",
                "|---|---|---|---|---|",
            ]
            for cs in lbl_cols:
                vr = cs.value_range
                if "classes" in vr:
                    vr_str = "Classes: " + ", ".join(str(c) for c in vr["classes"])
                elif "min" in vr:
                    vr_str = f"[{vr['min']:.4g}, {vr['max']:.4g}]"
                else:
                    vr_str = "n/a"
                lines.append(
                    f"| `{cs.name}` | {cs.dtype} | {cs.nan_rate:.1%} | {cs.n_valid:,} | {vr_str} |"
                )
            lines.append("")

        lines += ["---", ""]

        # Class distributions
        if meta.class_distributions:
            lines += ["## Class Distributions", ""]
            for col, dist in meta.class_distributions.items():
                if col not in meta.label_columns:
                    continue
                total = sum(dist.values())
                lines += [
                    f"### `{col}`",
                    "",
                    "| Class | Count | Proportion |",
                    "|---|---|---|",
                ]
                for cls, cnt in sorted(dist.items()):
                    lines.append(f"| {cls} | {cnt:,} | {cnt/max(total,1):.2%} |")
                lines.append("")
            lines += ["---", ""]

        # Missing values
        high_nan = [cs for cs in meta.column_summaries if cs.nan_rate > 0.01]
        if high_nan:
            lines += [
                "## Missing Values",
                "",
                "Columns with > 1 % missing values:",
                "",
                "| Column | NaN Count | NaN Rate |",
                "|---|---|---|",
            ]
            for cs in sorted(high_nan, key=lambda c: -c.nan_rate):
                lines.append(f"| `{cs.name}` | {cs.nan_count:,} | {cs.nan_rate:.2%} |")
            lines.append("")
            lines += ["---", ""]

        # Validation
        lines += [
            "## Validation Results",
            "",
            f"```",
            meta.validation_summary,
            "```",
            "",
            "---",
            "",
        ]

        # Selected features (full list)
        lines += [
            "## Selected Features",
            "",
            f"Total: **{meta.feature_count}** features",
            "",
            "```",
        ]
        for col in meta.feature_columns:
            lines.append(col)
        lines += ["```", "", "---", ""]

        # Selected labels
        lines += [
            "## Selected Labels",
            "",
            f"Total: **{meta.label_count}** label columns  ",
            f"Groups: {', '.join(meta.label_groups) if meta.label_groups else 'all'}",
            "",
            "```",
        ]
        for col in meta.label_columns:
            lines.append(col)
        lines += ["```", ""]

        # Footer
        lines += [
            "---",
            "",
            "> **Important**: Labels are strictly forward-looking targets.",
            "> Never use label columns as model input features.",
            "> Always use time-series cross-validation (walk-forward) to avoid look-ahead bias.",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("DatasetReport written → %s", path)
        return path
