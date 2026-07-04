"""Rolling window statistics on the close price."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_WINDOW = 20

_ROLLING_COLS: list[str] = [
    "rolling_mean",
    "rolling_median",
    "rolling_var",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_q25",
    "rolling_q75",
    "rolling_mad",
]


@FeatureRegistry.register
class RollingStatisticsEngine(BaseFeature):
    """20-bar rolling mean, median, variance, std, min, max, Q25, Q75, MAD on close."""

    name:             str       = "rolling_statistics"
    category:         str       = "statistics"
    dependencies:     list[str] = []
    required_columns: list[str] = ["close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        close_s = df["close"].astype(np.float64)
        roll    = close_s.rolling(_WINDOW, min_periods=1)

        rolling_mean   = roll.mean().to_numpy()
        rolling_median = roll.median().to_numpy()
        rolling_var    = roll.var(ddof=0).to_numpy()
        rolling_std    = roll.std(ddof=0).to_numpy()
        rolling_min    = roll.min().to_numpy()
        rolling_max    = roll.max().to_numpy()
        rolling_q25    = roll.quantile(0.25).to_numpy()
        rolling_q75    = roll.quantile(0.75).to_numpy()
        rolling_mad    = roll.apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        ).to_numpy()

        out = pd.DataFrame(index=df.index)
        out["rolling_mean"]   = rolling_mean
        out["rolling_median"] = rolling_median
        out["rolling_var"]    = rolling_var
        out["rolling_std"]    = rolling_std
        out["rolling_min"]    = rolling_min
        out["rolling_max"]    = rolling_max
        out["rolling_q25"]    = rolling_q25
        out["rolling_q75"]    = rolling_q75
        out["rolling_mad"]    = rolling_mad
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "20-bar rolling statistics on close: mean, median, variance (ddof=0), "
                "std, min, max, 25th/75th percentile, and mean absolute deviation. "
                "9 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _ROLLING_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "low",
            tags       = ["rolling", "mean", "std", "quantile", "mad", "statistics"],
        )
