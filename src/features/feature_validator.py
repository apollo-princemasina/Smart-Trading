"""FeatureValidator — deep quality checks on feature generator output.

Complements ``BaseFeature.validate_output()`` (structural contract) with
statistical and leakage-focused checks.  The pipeline runs the validator
after every ``generate()`` call and collects the results into the pipeline
report.

Checks performed
----------------
1.  **Row count match**     — output rows == input rows.
2.  **Index alignment**     — output.index identical to input.index.
3.  **NaN detection**       — count of NaN cells per output column.
4.  **Infinite values**     — count of ±Inf cells.
5.  **Duplicate columns**   — column names repeated in output.
6.  **Constant columns**    — columns with zero or near-zero variance.
7.  **Object dtype columns** — string / mixed-type columns (unsupported by ML).
8.  **Leakage candidates**  — columns whose names suggest future information
                              (suffix ``_future``, prefix ``next_``).
9.  **Unexpected index change** — index dtype or timezone changed.
10. **Column count sanity** — at least one output column if feature is enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base_feature import BaseFeature

logger = logging.getLogger(__name__)

# Naming patterns that indicate a column may carry future price information.
_LEAKAGE_PREFIXES = ("next_", "future_", "fwd_", "lookahead_")
_LEAKAGE_SUFFIXES = ("_future", "_next", "_fwd", "_lookahead", "_t1", "_t+1")


@dataclass
class FeatureValidationReport:
    """Validation result for one feature generator's output.

    Attributes
    ----------
    feature_name:
        The ``BaseFeature.name`` of the generator being validated.
    passed:
        ``True`` only when *all* hard checks pass (no issues).
    row_count_match:
        Output has the same number of rows as the input.
    index_aligned:
        Output index is identical to the input index.
    nan_count:
        Total NaN cells across all output columns.
    inf_count:
        Total ±Inf cells across numeric output columns.
    duplicate_columns:
        List of column names that appear more than once in the output.
    constant_columns:
        List of numeric column names with near-zero standard deviation.
    object_dtype_columns:
        List of columns with ``object`` dtype (strings / mixed types).
    potential_leakage_columns:
        Columns whose names match known look-ahead naming patterns.
    output_column_count:
        Total number of columns in the output DataFrame.
    issues:
        Hard failures that cause ``passed`` to be ``False``.
    warnings:
        Non-fatal observations logged to the report.
    """

    feature_name: str
    passed:       bool = True

    row_count_match:           bool       = True
    index_aligned:             bool       = True
    nan_count:                 int        = 0
    inf_count:                 int        = 0
    duplicate_columns:         list[str]  = field(default_factory=list)
    constant_columns:          list[str]  = field(default_factory=list)
    object_dtype_columns:      list[str]  = field(default_factory=list)
    potential_leakage_columns: list[str]  = field(default_factory=list)
    output_column_count:       int        = 0

    issues:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def _fail(self, message: str) -> None:
        self.passed = False
        self.issues.append(message)

    def _warn(self, message: str) -> None:
        self.warnings.append(message)


class FeatureValidator:
    """Run all validation checks on a feature generator's output.

    Usage::

        validator = FeatureValidator()
        report = validator.validate(input_df, output_df, feature_instance)
    """

    def validate(
        self,
        input_df:  pd.DataFrame,
        output_df: pd.DataFrame,
        feature:   BaseFeature,
    ) -> FeatureValidationReport:
        """Execute all checks and return a consolidated report.

        Parameters
        ----------
        input_df:
            The merged OHLCV DataFrame passed to ``feature.generate()``.
        output_df:
            The DataFrame returned by ``feature.generate()``.
        feature:
            The feature instance that produced *output_df*.

        Returns
        -------
        FeatureValidationReport
            Always returned (never raises).  Check ``report.passed``.
        """
        report = FeatureValidationReport(feature_name=feature.name)
        report.output_column_count = len(output_df.columns)

        self._check_row_count(input_df, output_df, report)
        self._check_index_alignment(input_df, output_df, report)
        self._check_duplicate_columns(output_df, report)
        self._check_object_dtype_columns(output_df, report)
        self._check_nan_values(output_df, report)
        self._check_infinite_values(output_df, report)
        self._check_constant_columns(output_df, report)
        self._check_leakage_column_names(output_df, report)
        self._check_empty_output(output_df, feature, report)

        return report

    # ── Individual checks ────────────────────────────────────────────────────

    @staticmethod
    def _check_row_count(
        input_df:  pd.DataFrame,
        output_df: pd.DataFrame,
        report:    FeatureValidationReport,
    ) -> None:
        if len(output_df) != len(input_df):
            report.row_count_match = False
            report._fail(
                f"Row count mismatch: input has {len(input_df)} rows, "
                f"output has {len(output_df)} rows."
            )

    @staticmethod
    def _check_index_alignment(
        input_df:  pd.DataFrame,
        output_df: pd.DataFrame,
        report:    FeatureValidationReport,
    ) -> None:
        if not output_df.index.equals(input_df.index):
            report.index_aligned = False
            report._fail(
                "Output index does not match input index. "
                "Ensure generate() preserves the original DataFrame index."
            )

        # Detect timezone changes (could indicate accidental tz-conversion)
        in_tz  = getattr(input_df.index.dtype,  "tz", None)
        out_tz = getattr(output_df.index.dtype, "tz", None)
        if in_tz != out_tz:
            report._warn(
                f"Index timezone changed: input={in_tz}, output={out_tz}. "
                "This may cause alignment issues downstream."
            )

    @staticmethod
    def _check_duplicate_columns(
        output_df: pd.DataFrame,
        report:    FeatureValidationReport,
    ) -> None:
        dups = output_df.columns[output_df.columns.duplicated()].tolist()
        report.duplicate_columns = dups
        if dups:
            report._fail(
                f"Duplicate column names in output: {dups}. "
                "Use unique column names or a unique feature prefix."
            )

    @staticmethod
    def _check_object_dtype_columns(
        output_df: pd.DataFrame,
        report:    FeatureValidationReport,
    ) -> None:
        obj_cols = [c for c in output_df.columns if output_df[c].dtype == object]
        report.object_dtype_columns = obj_cols
        if obj_cols:
            report._fail(
                f"Object-dtype columns in output: {obj_cols}. "
                "All feature columns must be numeric (int or float). "
                "Encode categorical values before returning."
            )

    @staticmethod
    def _check_nan_values(
        output_df: pd.DataFrame,
        report:    FeatureValidationReport,
    ) -> None:
        total_nan = int(output_df.isnull().sum().sum())
        report.nan_count = total_nan
        if total_nan:
            per_col = output_df.isnull().sum()
            worst = per_col[per_col > 0].sort_values(ascending=False).head(5).to_dict()
            report._warn(
                f"{total_nan} NaN values found across output columns. "
                f"Top offenders: {worst}. "
                "NaNs are tolerated for warm-up periods but must be handled "
                "before ML training."
            )

    @staticmethod
    def _check_infinite_values(
        output_df: pd.DataFrame,
        report:    FeatureValidationReport,
    ) -> None:
        numeric = output_df.select_dtypes(include=[float, int])
        if numeric.empty:
            return
        inf_mask = np.isinf(numeric.values)
        total_inf = int(inf_mask.sum())
        report.inf_count = total_inf
        if total_inf:
            report._fail(
                f"{total_inf} ±Inf values in numeric output columns. "
                "Infinite values break all downstream ML operations. "
                "Guard against division-by-zero in generate()."
            )

    @staticmethod
    def _check_constant_columns(
        output_df: pd.DataFrame,
        report:    FeatureValidationReport,
        tol: float = 1e-10,
    ) -> None:
        numeric = output_df.select_dtypes(include=[float, int])
        const_cols = [
            col for col in numeric.columns
            if numeric[col].std() < tol
        ]
        report.constant_columns = const_cols
        if const_cols:
            report._warn(
                f"Constant (zero-variance) columns: {const_cols}. "
                "These carry no information and will be removed by feature selection."
            )

    @staticmethod
    def _check_leakage_column_names(
        output_df: pd.DataFrame,
        report:    FeatureValidationReport,
    ) -> None:
        leaky = [
            col for col in output_df.columns
            if any(col.startswith(p) for p in _LEAKAGE_PREFIXES)
            or any(col.endswith(s) for s in _LEAKAGE_SUFFIXES)
        ]
        report.potential_leakage_columns = leaky
        if leaky:
            report._warn(
                f"Potential data-leakage column names: {leaky}. "
                "Column names suggest future information. "
                "Verify these are truly label columns (in the 'labels' category), "
                "not accidental lookahead."
            )

    @staticmethod
    def _check_empty_output(
        output_df: pd.DataFrame,
        feature:   BaseFeature,
        report:    FeatureValidationReport,
    ) -> None:
        if output_df.empty or output_df.shape[1] == 0:
            if feature.category != "labels":
                # Placeholder features intentionally return empty DataFrames.
                # Only warn — the pipeline handles empty outputs gracefully.
                report._warn(
                    f"generate() returned an empty DataFrame (0 columns). "
                    f"This is expected for placeholder features; "
                    "implement compute logic to produce real columns."
                )


# ── Aggregate validation across all features ─────────────────────────────────


@dataclass
class PipelineValidationSummary:
    """Rolled-up validation results for the entire pipeline run.

    Attributes
    ----------
    total_features:
        Number of feature generators that executed.
    passed_count:
        Generators with ``FeatureValidationReport.passed == True``.
    failed_count:
        Generators with at least one hard failure.
    total_nan_cells:
        Sum of NaN cells across all feature outputs.
    total_inf_cells:
        Sum of ±Inf cells across all feature outputs.
    failed_features:
        Names of generators that failed validation.
    """

    total_features: int = 0
    passed_count:   int = 0
    failed_count:   int = 0
    total_nan_cells: int = 0
    total_inf_cells: int = 0
    failed_features: list[str] = field(default_factory=list)
    all_reports:     list[FeatureValidationReport] = field(default_factory=list)

    @classmethod
    def from_reports(
        cls, reports: list[FeatureValidationReport]
    ) -> "PipelineValidationSummary":
        """Build a summary from a list of per-feature reports."""
        summary = cls()
        summary.total_features  = len(reports)
        summary.all_reports     = reports
        summary.passed_count    = sum(1 for r in reports if r.passed)
        summary.failed_count    = summary.total_features - summary.passed_count
        summary.total_nan_cells = sum(r.nan_count for r in reports)
        summary.total_inf_cells = sum(r.inf_count for r in reports)
        summary.failed_features = [r.feature_name for r in reports if not r.passed]
        return summary

    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0
