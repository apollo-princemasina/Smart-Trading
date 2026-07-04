"""
Dataset Validator
=================
Validates the assembled training dataset before it is saved to disk.

Checks performed
----------------
1.  empty             — Dataset has at least one row and one column.
2.  time_ordering     — Index is monotonically increasing (no gaps detected).
3.  duplicate_rows    — No exact-duplicate index values.
4.  duplicate_cols    — No duplicate column names.
5.  min_rows          — Row count satisfies the configured minimum.
6.  feature_nan_rate  — No feature column exceeds ``max_feature_nan_rate``.
7.  label_nan_rate    — No label column has ALL values NaN.
8.  target_available  — The primary target column (if specified) has valid data.
9.  dtype_consistency — All feature columns are numeric; labels are numeric.
10. schema_columns    — All expected columns are present.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PASS    = "PASS"
WARNING = "WARNING"
FAIL    = "FAIL"


@dataclass
class ValidationIssue:
    severity: str     # PASS / WARNING / FAIL
    check:    str
    message:  str


@dataclass
class DatasetValidationReport:
    passed:           bool
    issues:           list[ValidationIssue] = field(default_factory=list)
    row_count_input:  int  = 0
    row_count_output: int  = 0
    dropped_rows:     int  = 0
    feature_count:    int  = 0
    label_count:      int  = 0
    summary:          str  = ""

    def warnings(self)  -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    def failures(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == FAIL]

    def __str__(self) -> str:
        lines = [f"DatasetValidation: {'PASSED' if self.passed else 'FAILED'} — {self.summary}"]
        for issue in self.issues:
            if issue.severity != PASS:
                lines.append(f"  [{issue.severity}] {issue.check}: {issue.message}")
        return "\n".join(lines)


@dataclass
class DatasetValidatorConfig:
    min_rows:              int   = 100
    max_feature_nan_rate:  float = 0.50
    max_label_nan_rate:    float = 1.00   # labels are allowed partial NaN
    check_dtype:           bool  = True
    expected_columns:      list[str] = field(default_factory=list)


class DatasetValidator:
    """Validate an assembled feature-label DataFrame."""

    def __init__(self, config: Optional[DatasetValidatorConfig] = None) -> None:
        self.config = config or DatasetValidatorConfig()

    # ------------------------------------------------------------------
    def validate(
        self,
        dataset:         pd.DataFrame,
        feature_columns: list[str],
        label_columns:   list[str],
        primary_target:  Optional[str] = None,
    ) -> DatasetValidationReport:
        issues: list[ValidationIssue] = []
        cfg = self.config

        self._check_empty(dataset, issues)
        self._check_time_ordering(dataset, issues)
        self._check_duplicate_rows(dataset, issues)
        self._check_duplicate_cols(dataset, issues)
        self._check_min_rows(dataset, issues, cfg.min_rows)
        self._check_feature_nan(dataset, feature_columns, issues, cfg.max_feature_nan_rate)
        self._check_label_nan(dataset, label_columns, issues)
        if primary_target:
            self._check_target(dataset, primary_target, issues)
        if cfg.check_dtype:
            self._check_dtypes(dataset, feature_columns, label_columns, issues)
        if cfg.expected_columns:
            self._check_expected_columns(dataset, cfg.expected_columns, issues)

        passed  = not any(i.severity == FAIL for i in issues)
        n_pass  = sum(i.severity == PASS    for i in issues)
        n_warn  = sum(i.severity == WARNING for i in issues)
        n_fail  = sum(i.severity == FAIL    for i in issues)
        summary = f"{n_pass} passed, {n_warn} warnings, {n_fail} failures"

        report = DatasetValidationReport(
            passed=passed,
            issues=issues,
            row_count_input=len(dataset),
            row_count_output=len(dataset),
            feature_count=len(feature_columns),
            label_count=len(label_columns),
            summary=summary,
        )
        if passed:
            logger.info("DatasetValidator: PASSED — %s", summary)
        else:
            logger.warning("DatasetValidator: FAILED — %s", summary)
        return report

    # ------------------------------------------------------------------
    def _check_empty(self, df: pd.DataFrame, issues: list) -> None:
        if df.empty:
            issues.append(ValidationIssue(FAIL, "empty", "Dataset is empty."))
        else:
            issues.append(ValidationIssue(PASS, "empty", f"{len(df):,} rows, {len(df.columns)} columns."))

    def _check_time_ordering(self, df: pd.DataFrame, issues: list) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(ValidationIssue(WARNING, "time_ordering",
                                          "Index is not a DatetimeIndex."))
            return
        if not df.index.is_monotonic_increasing:
            issues.append(ValidationIssue(FAIL, "time_ordering",
                                          "Index is NOT monotonically increasing."))
        else:
            issues.append(ValidationIssue(PASS, "time_ordering",
                                          "Index is monotonically increasing."))

    def _check_duplicate_rows(self, df: pd.DataFrame, issues: list) -> None:
        n_dup = int(df.index.duplicated().sum())
        if n_dup > 0:
            issues.append(ValidationIssue(FAIL, "duplicate_rows",
                                          f"{n_dup} duplicate timestamp(s) found."))
        else:
            issues.append(ValidationIssue(PASS, "duplicate_rows",
                                          "No duplicate timestamps."))

    def _check_duplicate_cols(self, df: pd.DataFrame, issues: list) -> None:
        dup = [c for c in df.columns if list(df.columns).count(c) > 1]
        if dup:
            issues.append(ValidationIssue(FAIL, "duplicate_cols",
                                          f"Duplicate column names: {list(set(dup))[:5]}"))
        else:
            issues.append(ValidationIssue(PASS, "duplicate_cols",
                                          "No duplicate column names."))

    def _check_min_rows(self, df: pd.DataFrame, issues: list, min_rows: int) -> None:
        if len(df) < min_rows:
            issues.append(ValidationIssue(FAIL, "min_rows",
                                          f"Only {len(df)} rows; minimum is {min_rows}."))
        else:
            issues.append(ValidationIssue(PASS, "min_rows",
                                          f"{len(df):,} rows ≥ minimum {min_rows}."))

    def _check_feature_nan(
        self, df: pd.DataFrame, feature_cols: list[str], issues: list, max_rate: float
    ) -> None:
        for col in feature_cols:
            if col not in df.columns:
                continue
            rate = float(df[col].isna().mean())
            if rate > max_rate:
                issues.append(ValidationIssue(WARNING, "feature_nan_rate",
                                              f"Feature '{col}': {rate:.1%} NaN > {max_rate:.1%}."))

    def _check_label_nan(
        self, df: pd.DataFrame, label_cols: list[str], issues: list
    ) -> None:
        for col in label_cols:
            if col not in df.columns:
                continue
            col_data = df[col]
            # Duplicate column names return a DataFrame — take first occurrence
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            valid = int(col_data.notna().sum())
            nan_rate = float(col_data.isna().mean())
            if valid == 0:
                issues.append(ValidationIssue(FAIL, "label_nan_rate",
                                              f"Label '{col}' has ALL values NaN."))
            elif nan_rate > 0.80:
                issues.append(ValidationIssue(WARNING, "label_nan_rate",
                                              f"Label '{col}': {nan_rate:.1%} NaN."))

    def _check_target(
        self, df: pd.DataFrame, target: str, issues: list
    ) -> None:
        if target not in df.columns:
            issues.append(ValidationIssue(FAIL, "target_available",
                                          f"Primary target '{target}' not found in dataset."))
            return
        valid = int(df[target].notna().sum())
        total = len(df)
        if valid == 0:
            issues.append(ValidationIssue(FAIL, "target_available",
                                          f"Primary target '{target}' has no valid values."))
        else:
            issues.append(ValidationIssue(PASS, "target_available",
                                          f"Target '{target}': {valid:,}/{total:,} valid rows."))

    def _check_dtypes(
        self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str], issues: list
    ) -> None:
        non_numeric_feats = [
            c for c in feature_cols
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric_feats:
            issues.append(ValidationIssue(WARNING, "dtype_consistency",
                                          f"Non-numeric features: {non_numeric_feats[:5]}"))
        else:
            issues.append(ValidationIssue(PASS, "dtype_consistency",
                                          "All feature columns are numeric."))

    def _check_expected_columns(
        self, df: pd.DataFrame, expected: list[str], issues: list
    ) -> None:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            issues.append(ValidationIssue(FAIL, "schema_columns",
                                          f"Expected columns missing: {missing[:5]}"))
        else:
            issues.append(ValidationIssue(PASS, "schema_columns",
                                          f"All {len(expected)} expected columns present."))
