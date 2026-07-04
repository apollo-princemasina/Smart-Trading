"""Placeholder feature generator for the liquidity category.

Future generators in this package will detect:
- Liquidity pools (equal highs / equal lows)
- Stop-hunt sweeps above/below key levels
- Buy-side and sell-side liquidity zones
- HexaTrades Liquidity Sweep patterns
- Liquidity Magnet zones
"""

from __future__ import annotations

import pandas as pd

from ..base_feature import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class LiquidityPlaceholder(BaseFeature):
    """Placeholder: will be replaced by liquidity pool and sweep generators."""

    name             = "liquidity_placeholder"
    category         = "liquidity"
    dependencies: list[str]     = []
    required_columns: list[str] = []

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return empty DataFrame — no calculations yet."""
        return pd.DataFrame(index=df.index)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Placeholder for liquidity features: equal highs/lows detection, "
                "buy-side / sell-side liquidity pools, stop-hunt sweeps, "
                "and liquidity magnet zones."
            ),
            dependencies     = [],
            required_columns = [],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "high",
            tags             = ["ICT", "smart_money", "liquidity", "placeholder"],
        )
