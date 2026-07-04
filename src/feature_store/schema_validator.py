"""DataFrame validation against a frozen DatasetSchema."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .exceptions import ValidationError
from .feature_prefix import extract_prefix, is_valid_prefix
from .schema import DatasetSchema


# ── Validation report ─────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    """Full result of a schema validation pass."""

    is_valid:           bool
    errors:             list[str]               = field(default_factory=list)
    warnings:           list[str]               = field(default_factory=list)
    missing_features:   list[str]               = field(default_factory=list)
    extra_features:     list[str]               = field(default_factory=list)
    dtype_mismatches:   dict[str, tuple[str, str]] = field(default_factory=dict)
    prefix_violations:  list[str]               = field(default_factory=list)
    null_violations:    list[str]               = field(default_factory=list)
    range_violations:   dict[str, dict]         = field(default_factory=dict)
    duplicate_columns:  list[str]               = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise ValidationError(
                f"Schema validation failed with {len(self.errors)} error(s).",
                errors=self.errors,
            )

    def summary(self) -> str:
        lines = [
            f"Valid: {self.is_valid}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            f"Missing: {len(self.missing_features)}",
            f"Extra: {len(self.extra_features)}",
        ]
        if self.errors:
            lines += [f"  • {e}" for e in self.errors[:10]]
        return "\n".join(lines)


# ── Validator ─────────────────────────────────────────────────────────────────


class SchemaValidator:
    """Validate a pandas DataFrame against a ``DatasetSchema``."""

    def validate(
        self,
        df: pd.DataFrame,
        schema: DatasetSchema,
        strict: bool = True,
        check_ranges: bool = False,
    ) -> ValidationReport:
        """
        Validate *df* against *schema*.

        Args:
            df:            DataFrame to check.
            schema:        Reference schema.
            strict:        Treat extra columns as errors (not just warnings).
            check_ranges:  Verify min/max value bounds (expensive on large DFs).

        Returns:
            :class:`ValidationReport`
        """
        report = ValidationReport(is_valid=True)
        df_cols = list(df.columns)

        # ── Duplicate columns ─────────────────────────────────────────────────
        seen: set[str] = set()
        for col in df_cols:
            if col in seen:
                report.duplicate_columns.append(col)
            seen.add(col)
        if report.duplicate_columns:
            report.add_error(
                f"Duplicate columns: {report.duplicate_columns}"
            )

        df_col_set = set(df_cols)
        schema_col_set = set(schema.features)

        # ── Missing features ──────────────────────────────────────────────────
        report.missing_features = sorted(schema_col_set - df_col_set)
        for feat in report.missing_features:
            report.add_error(f"Missing required feature: '{feat}'")

        # ── Extra features ────────────────────────────────────────────────────
        report.extra_features = sorted(df_col_set - schema_col_set)
        for feat in report.extra_features:
            if strict:
                report.add_error(f"Unexpected feature: '{feat}'")
            else:
                report.add_warning(f"Unexpected feature (ignored): '{feat}'")

        # ── Dtype, prefix, nullability, and range checks ──────────────────────
        for feat_name, feat_schema in schema.features.items():
            if feat_name not in df_col_set:
                continue

            series = df[feat_name]

            # Dtype check (normalise pandas dtype to base type name)
            actual_dtype = str(series.dtype)
            expected_dtype = feat_schema.dtype
            if not _dtypes_compatible(actual_dtype, expected_dtype):
                report.dtype_mismatches[feat_name] = (expected_dtype, actual_dtype)
                report.add_error(
                    f"Dtype mismatch for '{feat_name}': "
                    f"expected '{expected_dtype}', got '{actual_dtype}'"
                )

            # Prefix check
            actual_prefix = extract_prefix(feat_name)
            if actual_prefix != feat_schema.prefix:
                report.prefix_violations.append(feat_name)
                report.add_error(
                    f"Prefix mismatch for '{feat_name}': "
                    f"schema says '{feat_schema.prefix}', "
                    f"extracted '{actual_prefix}'"
                )

            # Nullability check
            if not feat_schema.nullable and series.isna().any():
                report.null_violations.append(feat_name)
                report.add_error(
                    f"Null values in non-nullable feature '{feat_name}'"
                )

            # Range check (opt-in — expensive)
            if check_ranges:
                _check_range(feat_name, series, feat_schema, report)

        # ── Invalid prefixes (features in schema with invalid prefixes) ────────
        for feat_name in schema.features:
            if not is_valid_prefix(feat_name):
                report.prefix_violations.append(feat_name)
                report.add_error(
                    f"Feature '{feat_name}' has an invalid prefix"
                )

        return report

    def validate_or_raise(
        self,
        df: pd.DataFrame,
        schema: DatasetSchema,
        strict: bool = True,
    ) -> None:
        """Validate and raise :class:`ValidationError` if invalid."""
        report = self.validate(df, schema, strict=strict)
        report.raise_if_invalid()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dtypes_compatible(actual: str, expected: str) -> bool:
    """Loose dtype compatibility: float64 matches float32, int64 matches int32, etc."""
    def base(dt: str) -> str:
        dt = dt.lower()
        if "float" in dt:
            return "float"
        if "int" in dt:
            return "int"
        if "bool" in dt:
            return "bool"
        if "datetime" in dt:
            return "datetime"
        if "object" in dt or "str" in dt:
            return "object"
        return dt

    return base(actual) == base(expected)


def _check_range(
    feat_name: str,
    series: pd.Series,
    feat_schema,
    report: ValidationReport,
) -> None:
    numeric = pd.api.types.is_numeric_dtype(series)
    if not numeric:
        return
    if feat_schema.min_value is not None:
        n_below = int((series < feat_schema.min_value).sum())
        if n_below:
            report.range_violations[feat_name] = {
                "type": "below_min",
                "min_value": feat_schema.min_value,
                "n_violations": n_below,
            }
            report.add_warning(
                f"'{feat_name}': {n_below} values below min "
                f"({feat_schema.min_value})"
            )
    if feat_schema.max_value is not None:
        n_above = int((series > feat_schema.max_value).sum())
        if n_above:
            report.range_violations.setdefault(feat_name, {}).update(
                {"type": "above_max", "max_value": feat_schema.max_value,
                 "n_violations": n_above}
            )
            report.add_warning(
                f"'{feat_name}': {n_above} values above max "
                f"({feat_schema.max_value})"
            )
