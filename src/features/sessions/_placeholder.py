"""Placeholder feature generator for the sessions category.

Future generators will produce:
- London / New York / Asia / Pacific session boolean flags
- ICT Kill Zone markers (London Open, NY Open, etc.)
- Session high / low / open / close anchors
- Time-of-day sine/cosine encoding (cyclical time features)
- Day-of-week encoding
"""

from __future__ import annotations

import pandas as pd

from ..base_feature import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class SessionsPlaceholder(BaseFeature):
    """Placeholder: will be replaced by session-marker generators."""

    name             = "sessions_placeholder"
    category         = "sessions"
    dependencies: list[str]     = []
    required_columns: list[str] = ["timestamp"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return empty DataFrame — no calculations yet."""
        return pd.DataFrame(index=df.index)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Placeholder for session features: London/NY/Asia session flags, "
                "ICT Kill Zone markers, session OHLC anchors, "
                "and cyclical time-of-day / day-of-week encodings."
            ),
            dependencies     = [],
            required_columns = ["timestamp"],
            output_columns   = [],
            version          = "0.1.0",
            author           = "Smart Trading Team",
            complexity       = "low",
            tags             = ["ICT", "sessions", "time_features", "placeholder"],
        )
