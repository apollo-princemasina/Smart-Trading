"""Placeholder feature generator for the momentum category.

Future generators will compute:
- RSI (14, 21) and RSI divergence
- MACD (12/26/9 EMA) — line, signal, histogram
- Stochastic oscillator (%K, %D)
- ADX — trend strength
- Rate of change (ROC) — multiple periods
- Z-score normalised momentum
"""

from __future__ import annotations

import pandas as pd

from ..base_feature import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class MomentumPlaceholder(BaseFeature):
    """Placeholder: will be replaced by RSI, MACD, Stochastic, ADX generators."""

    name             = "momentum_placeholder"
    category         = "momentum"
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
                "Placeholder for momentum features: RSI, MACD, Stochastic, "
                "ADX, Rate of Change, and Z-score normalised momentum signals."
            ),
            dependencies     = [],
            required_columns = ["close"],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "low",
            tags             = ["momentum", "RSI", "MACD", "ADX", "placeholder"],
        )
