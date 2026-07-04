"""Per-feature-module contract: declared inputs, outputs, and datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import FeatureSchema
from .schema_validator import SchemaValidator, ValidationReport


@dataclass
class FeatureContract:
    """
    Contract declaration for a single feature-engineering module.

    Every feature engine should expose one of these so the Feature Store
    can validate its output before it is merged into the fused dataset.

    Fields
    ------
    module_name:
        Python class or module name (e.g. ``"TechnicalEngine"``).
    version:
        Module version string.
    input_features:
        Column names required as inputs (base OHLCV or upstream features).
    output_schemas:
        ``{col_name: FeatureSchema}`` — the contract for every output column.
    categories:
        Logical categories produced (e.g. ``["momentum", "trend"]``).
    timeframes:
        Timeframes this module operates on (e.g. ``["M15", "H1"]``).
    dependencies:
        Names of other feature modules that must run first.
    """

    module_name:      str
    version:          str                         = "1.0.0"
    input_features:   list[str]                   = field(default_factory=list)
    output_schemas:   dict[str, FeatureSchema]    = field(default_factory=dict)
    categories:       list[str]                   = field(default_factory=list)
    timeframes:       list[str]                   = field(default_factory=list)
    dependencies:     list[str]                   = field(default_factory=list)

    # ── Derived ────────────────────────────────────────────────────────────────

    @property
    def output_features(self) -> list[str]:
        return list(self.output_schemas)

    @property
    def output_dtypes(self) -> dict[str, str]:
        return {name: fs.dtype for name, fs in self.output_schemas.items()}

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_output(
        self,
        df: pd.DataFrame,
        strict: bool = True,
    ) -> ValidationReport:
        """
        Validate that *df* contains the declared output columns with
        correct dtypes and prefixes.

        Only columns declared in :attr:`output_schemas` are checked.  The
        input (OHLCV) columns in *df* are intentionally ignored.
        """
        from .schema import DatasetSchema

        # Build a minimal DatasetSchema covering only this module's outputs
        pseudo_schema = DatasetSchema(
            version   = self.version,
            symbol    = "__contract__",
            features  = {k: v for k, v in self.output_schemas.items()},
            created_at = "",
            frozen    = False,
            schema_hash = "",
        )

        # Subset df to only the declared outputs that exist
        existing_cols = [c for c in self.output_schemas if c in df.columns]
        validator = SchemaValidator()
        return validator.validate(df[existing_cols], pseudo_schema, strict=strict)

    def validate_input(self, df: pd.DataFrame) -> list[str]:
        """
        Return a list of missing input features (should be empty on success).
        """
        return [col for col in self.input_features if col not in df.columns]

    def to_dict(self) -> dict:
        return {
            "module_name":    self.module_name,
            "version":        self.version,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "categories":     self.categories,
            "timeframes":     self.timeframes,
            "dependencies":   self.dependencies,
            "output_schemas": {k: v.to_dict() for k, v in self.output_schemas.items()},
        }
