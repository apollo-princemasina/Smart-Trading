"""Placeholder — superseded by the Statistical Engine sub-modules."""

from __future__ import annotations

import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class StatisticsPlaceholder(BaseFeature):
    """Placeholder — replaced by all StatisticalEngine sub-modules."""

    name:             str       = "statistics_placeholder"
    category:         str       = "statistics"
    dependencies:     list[str] = []
    required_columns: list[str] = ["close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df.index)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = "Placeholder for statistical feature generators.",
            dependencies     = [],
            required_columns = ["close"],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "low",
            tags             = ["statistics", "placeholder"],
        )
