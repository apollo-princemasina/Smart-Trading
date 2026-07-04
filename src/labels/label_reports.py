"""
Label Reports
=============
Generates all label-related report artefacts:

  reports/labels/
    label_report.md           — comprehensive Markdown report
    label_metadata.json       — JSON metadata
    class_distribution.csv    — class counts per categorical label
    transition_matrix.csv     — direction transitions (t → t+1)
    confusion_baseline.csv    — trivial-classifier baseline accuracy
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .label_metadata import LabelMeta

logger = logging.getLogger(__name__)

_DEFAULT_REPORT_DIR = Path("reports") / "labels"


class LabelReportGenerator:
    """Generate all label reports from a label DataFrame + metadata."""

    def __init__(self, report_dir: Optional[Path] = None) -> None:
        self.report_dir = Path(report_dir or _DEFAULT_REPORT_DIR)

    # ------------------------------------------------------------------
    def generate_all(
        self,
        labels:   pd.DataFrame,
        metadata: LabelMeta,
    ) -> dict[str, Path]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        paths["metadata"]       = self._write_metadata(metadata)
        paths["report_md"]      = self._write_markdown(labels, metadata)
        paths["class_dist"]     = self._write_class_distribution(labels)
        paths["transition"]     = self._write_transition_matrix(labels)
        paths["confusion_base"] = self._write_confusion_baseline(labels)

        logger.info("LabelReports: wrote %d artefacts to %s", len(paths), self.report_dir)
        return paths

    # ------------------------------------------------------------------
    def _write_metadata(self, meta: LabelMeta) -> Path:
        path = self.report_dir / "label_metadata.json"
        meta.to_json(path)
        return path

    # ------------------------------------------------------------------
    def _write_class_distribution(self, labels: pd.DataFrame) -> Path:
        path = self.report_dir / "class_distribution.csv"
        rows = []
        for col in labels.columns:
            valid = labels[col].dropna()
            if valid.nunique() <= 15:
                for val, cnt in valid.value_counts().items():
                    rows.append({
                        "label_column": col,
                        "class":        val,
                        "count":        cnt,
                        "proportion":   cnt / max(len(valid), 1),
                    })
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            pd.DataFrame(columns=["label_column", "class", "count", "proportion"]).to_csv(
                path, index=False
            )
        return path

    # ------------------------------------------------------------------
    def _write_transition_matrix(self, labels: pd.DataFrame) -> Path:
        path = self.report_dir / "transition_matrix.csv"
        # Find direction columns (ternary 0/1/2 with prefix direction_)
        dir_cols = [c for c in labels.columns if c.startswith("direction_")]
        if not dir_cols:
            dir_cols = [c for c in labels.columns if labels[c].dropna().nunique() <= 5]

        frames = []
        for col in dir_cols[:3]:  # limit to first 3
            s     = labels[col].dropna().astype(int)
            if len(s) < 2:
                continue
            trans = pd.crosstab(s.iloc[:-1].values, s.iloc[1:].values,
                                rownames=["from"], colnames=["to"], normalize="index")
            trans.index   = [f"{col}_{i}" for i in trans.index]
            frames.append(trans)

        if frames:
            pd.concat(frames).to_csv(path)
        else:
            pd.DataFrame().to_csv(path)
        return path

    # ------------------------------------------------------------------
    def _write_confusion_baseline(self, labels: pd.DataFrame) -> Path:
        """Trivial-classifier (majority-class) baseline accuracy per label column."""
        path = self.report_dir / "confusion_baseline.csv"
        rows = []
        for col in labels.columns:
            valid = labels[col].dropna()
            if valid.nunique() <= 15 and len(valid) >= 10:
                majority_acc  = float(valid.value_counts(normalize=True).max())
                n_classes     = valid.nunique()
                random_acc    = 1.0 / n_classes if n_classes > 0 else 0.0
                rows.append({
                    "label_column":    col,
                    "majority_acc":    majority_acc,
                    "random_baseline": random_acc,
                    "n_classes":       n_classes,
                    "n_valid":         len(valid),
                })
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            pd.DataFrame(columns=["label_column", "majority_acc",
                                  "random_baseline", "n_classes", "n_valid"]).to_csv(
                path, index=False
            )
        return path

    # ------------------------------------------------------------------
    def _write_markdown(self, labels: pd.DataFrame, meta: LabelMeta) -> Path:
        path = self.report_dir / "label_report.md"
        lines: list[str] = []

        lines += [
            f"# Label Generation Report",
            f"",
            f"**Symbol**: {meta.symbol}  ",
            f"**Timeframe**: {meta.timeframe or 'n/a'}  ",
            f"**Version**: v{meta.label_version}  ",
            f"**Generated**: {meta.generated_at}  ",
            f"**Generator**: {meta.generator}  ",
            f"",
            f"---",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total rows | {meta.n_rows:,} |",
            f"| Valid rows | {meta.n_valid_rows:,} ({meta.n_valid_rows/max(meta.n_rows,1):.1%}) |",
            f"| Label columns | {len(meta.label_columns)} |",
            f"| Validation | {'✓ PASSED' if meta.validation_passed else '✗ FAILED'} |",
            f"",
            f"---",
            f"",
            f"## Label Columns",
            f"",
            f"| Column | Dtype | NaN Rate | Valid Rows | Range / Classes |",
            f"|---|---|---|---|---|",
        ]

        for cm in meta.column_meta:
            vr_str = (
                ", ".join(str(c) for c in cm.value_range.get("classes", []))
                if "classes" in cm.value_range
                else f"[{cm.value_range.get('min', 'n/a'):.4g}, "
                     f"{cm.value_range.get('max', 'n/a'):.4g}]"
                     if cm.value_range.get("min") is not None else "n/a"
            )
            lines.append(
                f"| {cm.name} | {cm.dtype} | {cm.nan_rate:.1%} | {cm.n_valid:,} | {vr_str} |"
            )

        lines += ["", "---", "", "## Class Distributions", ""]
        for col, dist in meta.class_distributions.items():
            total = sum(dist.values())
            lines.append(f"### {col}")
            lines.append("")
            lines.append("| Class | Count | Proportion |")
            lines.append("|---|---|---|")
            for cls, cnt in sorted(dist.items()):
                lines.append(f"| {cls} | {cnt:,} | {cnt/max(total,1):.2%} |")
            lines.append("")

        lines += [
            "---",
            "",
            "## Model Targets",
            "",
            "| Model | Recommended Target Columns |",
            "|---|---|",
            "| Market Bias Classifier | direction_1b, direction_3b, direction_5b, direction_10b |",
            "| Market Bias Regressor | fwd_return_1b, fwd_return_3b, fwd_return_5b |",
            "| Setup Quality | setup_quality, setup_score |",
            "| Entry Timing | entry_signal, is_optimal_entry |",
            "| Trade Outcome Classifier | long_outcome, short_outcome, outcome |",
            "| Trade Outcome Regressor | long_mfe_pct, long_mae_pct, realized_rr |",
            "| Trade Management | mgmt_strategy, mgmt_optimal_exit_bar |",
            "",
            "---",
            "",
            f"## Validation",
            f"",
            f"```",
            meta.validation_summary,
            f"```",
            "",
            "---",
            "",
            "## Notes",
            "",
            meta.notes or "_No notes._",
            "",
            "> Labels are strictly forward-looking. They must NEVER be used as input features.",
            "> Always drop NaN rows before training. Last N rows are NaN by design.",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("LabelReport written → %s", path)
        return path
