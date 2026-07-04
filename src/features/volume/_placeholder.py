"""Placeholder feature generator for the volume category.

The volume directory already contains stub implementations of delta_volume.py
and volume_profile.py.  When those are fully implemented to inherit from
BaseFeature and use @FeatureRegistry.register, this placeholder can be removed.

Future generators will compute:
- Delta Volume (aggressive buy volume - sell volume approximation)
- Volume Profile (POC, VAH, VAL per session)
- Cumulative Volume Delta (CVD)
- On-Balance Volume (OBV)
- Volume Z-score (relative to rolling mean)
- Premium & Discount Delta Volume (LuxAlgo style)
"""

from __future__ import annotations

import pandas as pd

from ..base_feature import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class VolumePlaceholder(BaseFeature):
    """Placeholder: will be replaced by Delta Volume and Volume Profile generators."""

    name             = "volume_placeholder"
    category         = "volume"
    dependencies: list[str]     = []
    required_columns: list[str] = ["close", "tick_volume"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return empty DataFrame — no calculations yet."""
        return pd.DataFrame(index=df.index)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Placeholder for volume features: Delta Volume, Volume Profile "
                "(POC/VAH/VAL), Cumulative Volume Delta (CVD), OBV, "
                "and Premium & Discount Delta Volume."
            ),
            dependencies     = [],
            required_columns = ["close", "tick_volume"],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "medium",
            tags             = ["volume", "CVD", "OBV", "delta_volume", "placeholder"],
        )
