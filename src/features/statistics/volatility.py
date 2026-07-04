"""Volatility statistics: realised vol, historical vol, regime, expansion/compression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_RV_SHORT    = 20     # realised volatility window
_RV_LONG     = 60     # historical volatility window
_ATR_ROLL    = 20     # rolling mean of ATR
_REGIME_WIN  = 60     # window for volatility percentile rank

_VOL_STAT_COLS: list[str] = [
    "realized_volatility",
    "historical_volatility",
    "volatility_expansion",
    "volatility_compression",
    "atr_ratio",
    "rolling_atr_20",
    "volatility_regime",
]


def _pct_rank_scalar(x: np.ndarray) -> float:
    """Percentile rank of last element in x, scaled to [0, 1]."""
    if len(x) < 2:
        return 0.5
    return float(np.sum(x < x[-1]) / (len(x) - 1))


@FeatureRegistry.register
class VolatilityStatisticsEngine(BaseFeature):
    """Realised vol, historical vol, expansion/compression, ATR ratio, vol regime."""

    name:             str       = "volatility_stats"
    category:         str       = "statistics"
    dependencies:     list[str] = ["returns", "volatility"]
    required_columns: list[str] = ["log_return", "atr"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = df["log_return"].to_numpy(dtype=np.float64)
        atr     = df["atr"].to_numpy(dtype=np.float64)
        lr_s    = pd.Series(log_ret, index=df.index)
        atr_s   = pd.Series(atr,     index=df.index)

        # ── Realised / historical volatility (annualised ×√252, assumes daily bars) ─
        rv_short = (lr_s.rolling(_RV_SHORT, min_periods=2).std(ddof=1) * np.sqrt(252)
                    ).to_numpy()
        rv_long  = (lr_s.rolling(_RV_LONG,  min_periods=2).std(ddof=1) * np.sqrt(252)
                    ).to_numpy()
        rv_short = np.where(np.isnan(rv_short), 0.0, rv_short)
        rv_long  = np.where(np.isnan(rv_long),  0.0, rv_long)

        # ── Expansion / compression: short-term vol relative to long-term ─────
        safe_long = np.where(rv_long > 0, rv_long, 1.0)
        volatility_expansion    = np.where(rv_long > 0, rv_short / safe_long, 1.0)
        volatility_compression  = np.where(rv_short > 0,
                                           rv_long / np.where(rv_short > 0, rv_short, 1.0),
                                           1.0)

        # ── ATR ratio: current ATR / rolling mean of ATR ─────────────────────
        rolling_atr_20 = atr_s.rolling(_ATR_ROLL, min_periods=1).mean().to_numpy()
        safe_roll_atr  = np.where(rolling_atr_20 > 0, rolling_atr_20, 1.0)
        atr_ratio      = np.where(rolling_atr_20 > 0, atr / safe_roll_atr, 1.0)

        # ── Volatility regime: percentile rank of RV_short in 60-bar window ──
        volatility_regime = lr_s.rolling(_REGIME_WIN, min_periods=4).apply(
            lambda x: _pct_rank_scalar(
                pd.Series(x).rolling(_RV_SHORT, min_periods=2).std(ddof=1).dropna().to_numpy()
            ), raw=False
        ).to_numpy()
        volatility_regime = np.where(np.isnan(volatility_regime), 0.5, volatility_regime)

        out = pd.DataFrame(index=df.index)
        out["realized_volatility"]   = rv_short
        out["historical_volatility"] = rv_long
        out["volatility_expansion"]  = volatility_expansion
        out["volatility_compression"]= volatility_compression
        out["atr_ratio"]             = atr_ratio
        out["rolling_atr_20"]        = rolling_atr_20
        out["volatility_regime"]     = volatility_regime
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Realised volatility (20-bar std × √252), historical volatility "
                "(60-bar), expansion (RV_short/RV_long), compression (inverse), "
                "ATR ratio (current/rolling-mean), rolling ATR mean (20), and "
                "volatility regime (0–1 percentile rank). 7 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _VOL_STAT_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = ["volatility", "realized", "regime", "atr", "statistics"],
        )
