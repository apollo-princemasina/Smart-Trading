"""Price return features: log, simple, rolling, and forward returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_ROLL_SHORT = 5
_ROLL_LONG  = 20

_RETURN_COLS: list[str] = [
    "log_return",
    "simple_return",
    "rolling_return_5",
    "rolling_return_20",
    "fwd_return_1",
]


@FeatureRegistry.register
class ReturnsEngine(BaseFeature):
    """Log returns, simple returns, rolling sums, and 1-bar forward return."""

    name:             str       = "returns"
    category:         str       = "statistics"
    dependencies:     list[str] = []
    required_columns: list[str] = ["close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        close    = df["close"].to_numpy(dtype=np.float64)
        prev_cls = np.concatenate([[close[0]], close[:-1]])

        safe_prev     = np.where(prev_cls > 0, prev_cls, 1.0)
        log_return    = np.where(prev_cls > 0,
                                 np.log(np.maximum(close / safe_prev, 1e-10)),
                                 0.0)
        simple_return = np.where(prev_cls > 0,
                                 (close - prev_cls) / safe_prev,
                                 0.0)

        lr_s = pd.Series(log_return, index=df.index)
        rolling_return_5  = lr_s.rolling(_ROLL_SHORT, min_periods=1).sum().to_numpy()
        rolling_return_20 = lr_s.rolling(_ROLL_LONG,  min_periods=1).sum().to_numpy()

        # 1-bar forward log return; last bar has no future → fill with 0.0
        fwd_return_1 = np.concatenate([log_return[1:], [0.0]])

        out = pd.DataFrame(index=df.index)
        out["log_return"]        = log_return
        out["simple_return"]     = simple_return
        out["rolling_return_5"]  = rolling_return_5
        out["rolling_return_20"] = rolling_return_20
        out["fwd_return_1"]      = fwd_return_1
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Log return, simple return, rolling log-return sums (5/20 bars), "
                "and 1-bar forward log return (last bar = 0.0).  5 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _RETURN_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "low",
            tags       = ["returns", "log_return", "simple_return", "statistics"],
        )
