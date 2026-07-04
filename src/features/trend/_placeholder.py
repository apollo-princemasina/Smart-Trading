"""Placeholder feature generator for the trend category.

Future generators will compute:
- EMA stack alignment (9/21/50/200 EMA)
- Higher-timeframe trend bias (H1, H4, D1 direction)
- Price relative to key moving averages
- Trend slope and acceleration
- LuxAlgo Smart Money Concepts trend signals
"""

from __future__ import annotations

import pandas as pd

from ..base_feature import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class TrendPlaceholder(BaseFeature):
    """Placeholder: will be replaced by EMA stack and trend-bias generators."""

    name             = "trend_placeholder"
    category         = "trend"
    dependencies: list[str]     = []
    required_columns: list[str] = ["close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return empty DataFrame — no calculations yet."""
        return pd.DataFrame(index=df.index)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Placeholder for trend features: EMA stack (9/21/50/200), "
                "higher-timeframe bias, trend direction encoding, "
                "price-relative-to-MA ratios, and LuxAlgo SMC trend signals."
            ),
            dependencies     = [],
            required_columns = ["close"],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "medium",
            tags             = ["trend", "EMA", "moving_average", "placeholder"],
        )
