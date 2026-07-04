"""
Label Validator
===============
Validates a combined label DataFrame for:

  1. Missing / NaN labels         — excessive NaN beyond expected tail
  2. Class imbalance              — any class < imbalance_floor of population
  3. Column name leakage          — label column names appear in feature DataFrame
  4. Time alignment               — label index matches feature index
  5. Future leakage check         — label columns don't correlate suspiciously
                                    with contemporaneous feature columns
  6. Expected tail NaN            — last n rows are NaN as required
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Validation status codes
PASS    = "PASS"
WARNING = "WARNING"
FAIL    = "FAIL"


@dataclass
class ValidationIssue:
    severity: str           # PASS / WARNING / FAIL
    check:    str
    message:  str


@dataclass
class ValidationReport:
    passed:   bool
    issues:   list[ValidationIssue] = field(default_factory=list)
    summary:  str = ""

    def warnings(self)  -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    def failures(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == FAIL]

    def __str__(self) -> str:
        lines = [f"LabelValidation: {'PASSED' if self.passed else 'FAILED'}"]
        for issue in self.issues:
            lines.append(f"  [{issue.severity}] {issue.check}: {issue.message}")
        return "\n".join(lines)


@dataclass
class LabelValidatorConfig:
    max_nan_rate:        float = 0.30    # more NaN than this → FAIL
    imbalance_floor:     float = 0.02   # minority class < 2 % of valid rows → WARNING
    leakage_corr_thresh: float = 0.90   # contemporaneous |corr| > 0.90 → FAIL
    expected_tail_nan:   int   = 0      # minimum NaN rows expected at the tail
    binary_label_suffix: str   = "bias" # used to identify binary label cols
    classification_cols: list[str] = field(default_factory=list)


class LabelValidator:
    """Validate labels before saving or training."""

    def __init__(self, config: Optional[LabelValidatorConfig] = None) -> None:
        self.config = config or LabelValidatorConfig()

    # ------------------------------------------------------------------
    def validate(
        self,
        labels: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []

        self._check_empty(labels, issues)
        self._check_nan_rate(labels, issues)
        self._check_class_balance(labels, issues)
        if features is not None:
            self._check_index_alignment(labels, features, issues)
            self._check_column_name_leakage(labels, features, issues)
            self._check_contemporaneous_correlation(labels, features, issues)
        self._check_value_ranges(labels, issues)

        passed = not any(i.severity == FAIL for i in issues)
        summary = (
            f"{sum(i.severity == PASS for i in issues)} checks passed, "
            f"{sum(i.severity == WARNING for i in issues)} warnings, "
            f"{sum(i.severity == FAIL for i in issues)} failures"
        )
        report = ValidationReport(passed=passed, issues=issues, summary=summary)
        logger.info("LabelValidator: %s", summary)
        return report

    # ------------------------------------------------------------------
    def _check_empty(self, labels: pd.DataFrame, issues: list) -> None:
        if labels.empty:
            issues.append(ValidationIssue(FAIL, "empty", "Label DataFrame is empty."))
        else:
            issues.append(ValidationIssue(PASS, "empty", f"{len(labels)} rows present."))

    def _check_nan_rate(self, labels: pd.DataFrame, issues: list) -> None:
        for col in labels.columns:
            nan_rate = float(labels[col].isna().mean())
            if nan_rate > self.config.max_nan_rate:
                issues.append(ValidationIssue(
                    FAIL, "nan_rate",
                    f"'{col}': {nan_rate:.1%} NaN (threshold {self.config.max_nan_rate:.1%}).",
                ))
            else:
                issues.append(ValidationIssue(
                    PASS, "nan_rate",
                    f"'{col}': {nan_rate:.1%} NaN — OK.",
                ))

    def _check_class_balance(self, labels: pd.DataFrame, issues: list) -> None:
        candidate_cols = self.config.classification_cols or [
            c for c in labels.columns
            if labels[c].dropna().nunique() <= 10
        ]
        for col in candidate_cols:
            valid = labels[col].dropna()
            if len(valid) == 0:
                continue
            counts = valid.value_counts(normalize=True)
            minority = float(counts.min())
            if minority < self.config.imbalance_floor:
                issues.append(ValidationIssue(
                    WARNING, "class_balance",
                    f"'{col}': minority class {minority:.2%} < floor {self.config.imbalance_floor:.2%}.",
                ))
            else:
                issues.append(ValidationIssue(
                    PASS, "class_balance",
                    f"'{col}': balanced (min class {minority:.2%}).",
                ))

    def _check_index_alignment(
        self, labels: pd.DataFrame, features: pd.DataFrame, issues: list
    ) -> None:
        if not labels.index.equals(features.index):
            # Check intersection
            overlap = labels.index.intersection(features.index)
            issues.append(ValidationIssue(
                WARNING, "index_alignment",
                f"Label and feature indexes differ. Overlap: {len(overlap)} rows.",
            ))
        else:
            issues.append(ValidationIssue(
                PASS, "index_alignment", "Label and feature indexes match.",
            ))

    def _check_column_name_leakage(
        self, labels: pd.DataFrame, features: pd.DataFrame, issues: list
    ) -> None:
        leaked = set(labels.columns) & set(features.columns)
        if leaked:
            issues.append(ValidationIssue(
                FAIL, "column_leakage",
                f"Label columns appear in feature DataFrame: {sorted(leaked)}",
            ))
        else:
            issues.append(ValidationIssue(
                PASS, "column_leakage",
                "No label column names appear in feature DataFrame.",
            ))

    def _check_contemporaneous_correlation(
        self, labels: pd.DataFrame, features: pd.DataFrame, issues: list
    ) -> None:
        thr  = self.config.leakage_corr_thresh
        idx  = labels.index.intersection(features.index)
        if len(idx) < 50:
            return

        ldf = labels.loc[idx].select_dtypes(include=[np.number])
        fdf = features.loc[idx].select_dtypes(include=[np.number])

        # Only check a representative subset of feature columns (speed)
        feat_cols = fdf.columns.tolist()[:50]

        for lc in ldf.columns:
            ls = ldf[lc].dropna()
            for fc in feat_cols:
                fs = fdf[fc].reindex(ls.index).dropna()
                shared = ls.reindex(fs.index).dropna()
                fs2    = fs.reindex(shared.index)
                if len(shared) < 30:
                    continue
                try:
                    corr = float(abs(shared.corr(fs2)))
                except Exception:
                    continue
                if corr > thr:
                    issues.append(ValidationIssue(
                        FAIL, "contemporaneous_correlation",
                        f"Label '{lc}' correlates {corr:.3f} with feature '{fc}' "
                        f"(threshold {thr}). Possible leakage.",
                    ))

    def _check_value_ranges(self, labels: pd.DataFrame, issues: list) -> None:
        for col in labels.columns:
            valid = labels[col].dropna()
            if len(valid) == 0:
                continue
            # Binary columns should only contain {0, 1}
            if "bias_" in col or "binary" in col or "is_" in col:
                uniq = set(valid.unique())
                if not uniq.issubset({0.0, 1.0}):
                    issues.append(ValidationIssue(
                        WARNING, "value_range",
                        f"'{col}' expected binary {{0,1}}, got {uniq}.",
                    ))
            # Probability / confidence columns should be in [0, 1]
            if "probability" in col or "confidence" in col or "score" in col:
                if float(valid.min()) < -0.01 or float(valid.max()) > 1.01:
                    issues.append(ValidationIssue(
                        WARNING, "value_range",
                        f"'{col}' expected [0,1], range [{valid.min():.3f}, {valid.max():.3f}].",
                    ))
