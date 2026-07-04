"""Dataset-level schema contract — what a downstream consumer requires."""

from __future__ import annotations

from dataclasses import dataclass, field

from .exceptions import ValidationError
from .schema import DatasetSchema
from .schema_validator import ValidationReport


@dataclass
class SchemaContract:
    """
    Declarative contract that a downstream module places on the feature dataset.

    Any downstream stage (model training, backtesting, inference) should define
    a :class:`SchemaContract` and call :meth:`validate` before loading data.

    Fields
    ------
    required_features:
        Column names that MUST be present.
    optional_features:
        Column names that MAY be present (absence is not an error).
    required_prefixes:
        All required features must start with one of these prefixes.
    required_categories:
        At least one feature from each listed category must be present.
    min_feature_count:
        Minimum total number of features in the schema.
    max_feature_count:
        Maximum total (``0`` = no limit).
    """

    required_features:  list[str] = field(default_factory=list)
    optional_features:  list[str] = field(default_factory=list)
    required_prefixes:  list[str] = field(default_factory=list)
    required_categories: list[str] = field(default_factory=list)
    min_feature_count:  int        = 0
    max_feature_count:  int        = 0   # 0 = unlimited
    name:               str        = ""  # human-readable contract name

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, schema: DatasetSchema) -> ValidationReport:
        """
        Validate *schema* against this contract.

        Returns a :class:`ValidationReport` (never raises — check
        ``report.is_valid``).
        """
        report = ValidationReport(is_valid=True)
        schema_names = set(schema.features)

        # Missing required features
        for feat in self.required_features:
            if feat not in schema_names:
                report.missing_features.append(feat)
                report.add_error(f"Contract violation: required feature '{feat}' missing")

        # Required prefix coverage
        if self.required_prefixes:
            for prefix in self.required_prefixes:
                covered = any(
                    f.startswith(prefix) for f in schema_names
                )
                if not covered:
                    report.add_error(
                        f"Contract violation: no feature with prefix '{prefix}' found"
                    )

        # Required category coverage
        if self.required_categories:
            schema_categories = schema.categories
            for cat in self.required_categories:
                if cat not in schema_categories:
                    report.add_error(
                        f"Contract violation: no feature with category '{cat}' found"
                    )

        # Feature count
        fc = schema.feature_count
        if self.min_feature_count and fc < self.min_feature_count:
            report.add_error(
                f"Contract violation: schema has {fc} features, "
                f"minimum required is {self.min_feature_count}"
            )
        if self.max_feature_count and fc > self.max_feature_count:
            report.add_warning(
                f"Schema has {fc} features, expected ≤ {self.max_feature_count}"
            )

        return report

    def validate_or_raise(self, schema: DatasetSchema) -> None:
        """Validate and raise :class:`ValidationError` if the contract is violated."""
        report = self.validate(schema)
        if not report.is_valid:
            raise ValidationError(
                f"Schema contract '{self.name}' failed.",
                errors=report.errors,
            )

    def to_dict(self) -> dict:
        return {
            "name":               self.name,
            "required_features":  self.required_features,
            "optional_features":  self.optional_features,
            "required_prefixes":  self.required_prefixes,
            "required_categories": self.required_categories,
            "min_feature_count":  self.min_feature_count,
            "max_feature_count":  self.max_feature_count,
        }
