"""Placeholder feature generator for the volatility category.

Future generators will compute:
- Average True Range (ATR) — period 14, 20
- ATR as percentage of price
- Bollinger Bands (width, %B, position)
- Historical volatility (rolling standard deviation of log returns)
- Volatility regime (low / normal / high)
"""

from __future__ import annotations

import pandas as pd

from ..base_feature import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class VolatilityPlaceholder(BaseFeature):
    """Placeholder: will be replaced by ATR and Bollinger generators."""

    name             = "volatility_placeholder"
    category         = "volatility"
    dependencies: list[str]     = []
    required_columns: list[str] = ["high", "low", "close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return empty DataFrame — no calculations yet."""
        return pd.DataFrame(index=df.index)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Placeholder for volatility features: ATR (14/20), "
                "Bollinger Band width and %B position, "
                "historical volatility (log-return std), "
                "and volatility regime classification."
            ),
            dependencies     = [],
            required_columns = ["high", "low", "close"],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "low",
            tags             = ["volatility", "ATR", "Bollinger", "placeholder"],
        )
