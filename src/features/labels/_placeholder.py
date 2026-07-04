"""Placeholder feature generator for the labels category.

This is the only category where output columns intentionally describe future
price outcomes (the prediction targets).  The validator allows 'next_' prefixed
columns in the 'labels' category without raising a leakage warning.

Future generators will produce:
- Triple Barrier labels (Lopez de Prado) — long / short / neutral
- Binary directional labels (up / down over N candles)
- Risk-reward constrained labels (only label when R:R >= threshold)
- Continuous return targets (regression targets for price prediction)
"""

from __future__ import annotations

import pandas as pd

from ..base_feature import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class LabelsPlaceholder(BaseFeature):
    """Placeholder: will be replaced by Triple Barrier and binary label generators."""

    name             = "labels_placeholder"
    category         = "labels"
    dependencies: list[str]     = []
    required_columns: list[str] = ["close", "high", "low"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return empty DataFrame — no label calculations yet."""
        return pd.DataFrame(index=df.index)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Placeholder for ML target labels: Triple Barrier (Lopez de Prado), "
                "binary directional labels, risk-reward filtered labels, "
                "and continuous return targets."
            ),
            dependencies     = [],
            required_columns = ["close", "high", "low"],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "medium",
            tags             = ["labels", "triple_barrier", "targets", "placeholder"],
        )
