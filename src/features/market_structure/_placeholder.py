"""Placeholder feature generator for the market_structure category.

This module registers a no-op generator that satisfies the BaseFeature
contract and proves the registry / pipeline wiring is correct.

When market-structure indicators are implemented (BOS, CHoCH, Order Blocks,
FVGs, Swing Highs/Lows, etc.), they will live alongside this file and follow
the same @FeatureRegistry.register pattern.  This file can then be removed or
kept as a smoke-test fixture.
"""

from __future__ import annotations

import pandas as pd

from ..base_feature import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class MarketStructurePlaceholder(BaseFeature):
    """Placeholder: will be replaced by BOS, CHoCH, Order Block generators."""

    name             = "market_structure_placeholder"
    category         = "market_structure"
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
                "Placeholder for market-structure features: Break of Structure (BOS), "
                "Change of Character (CHoCH), Order Blocks, Fair Value Gaps, "
                "Swing Highs/Lows, and Market Structure Shifts (MSS)."
            ),
            dependencies     = [],
            required_columns = [],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "high",
            tags             = ["ICT", "smart_money", "market_structure", "placeholder"],
        )
