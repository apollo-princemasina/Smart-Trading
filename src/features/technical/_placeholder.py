"""Placeholder feature generator for the technical category."""

from __future__ import annotations

import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class TechnicalPlaceholder(BaseFeature):
    """Placeholder — replaced by the Technical Indicator Engine sub-modules."""

    name:             str       = "technical_placeholder"
    category:         str       = "technical"
    dependencies:     list[str] = []
    required_columns: list[str] = ["close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df.index)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = "Placeholder for technical indicator features.",
            dependencies     = [],
            required_columns = ["close"],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "low",
            tags             = ["technical", "placeholder"],
        )
