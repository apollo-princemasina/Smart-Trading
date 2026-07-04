"""Composite statistical engine — cross-module derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_STAT_ENGINE_COLS: list[str] = [
    "return_vol_ratio",
    "trend_quality",
    "noise_ratio",
    "price_efficiency",
    "regime_consistency",
]


@FeatureRegistry.register
class StatisticalEngine(BaseFeature):
    """Cross-module composites combining returns, microstructure, and volatility signals."""

    name:             str       = "statistics"
    category:         str       = "statistics"
    dependencies:     list[str] = [
        "returns",
        "rolling_statistics",
        "distribution",
        "candle_statistics",
        "momentum_stats",
        "volatility_stats",
        "entropy",
        "market_microstructure",
    ]
    required_columns: list[str] = [
        "rolling_return_5", "rolling_std",
        "efficiency_ratio", "momentum_persistence",
        "market_noise", "volatility_regime",
        "hurst", "entropy",
        "volatility_expansion", "trend_persistence",
    ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_ret5   = df["rolling_return_5"].to_numpy(dtype=np.float64)
        rolling_std    = df["rolling_std"].to_numpy(dtype=np.float64)
        er             = df["efficiency_ratio"].to_numpy(dtype=np.float64)
        mom_persist    = df["momentum_persistence"].to_numpy(dtype=np.float64)
        market_noise   = df["market_noise"].to_numpy(dtype=np.float64)
        vol_regime     = df["volatility_regime"].to_numpy(dtype=np.float64)
        hurst          = df["hurst"].to_numpy(dtype=np.float64)
        entropy        = df["entropy"].to_numpy(dtype=np.float64)
        vol_expansion  = df["volatility_expansion"].to_numpy(dtype=np.float64)
        trend_persist  = df["trend_persistence"].to_numpy(dtype=np.float64)

        # ── return_vol_ratio: normalised return (Sharpe-like, no risk-free) ──
        safe_std = np.where(rolling_std > 0, rolling_std, 1.0)
        return_vol_ratio = np.where(rolling_std > 0,
                                    rolling_ret5 / safe_std, 0.0)

        # ── trend_quality: efficiency × persistence ───────────────────────────
        trend_quality = er * np.clip(mom_persist, 0.0, 1.0)

        # ── noise_ratio: market noise amplified by vol expansion ─────────────
        noise_ratio = market_noise * np.clip(vol_expansion, 0.5, 3.0)

        # ── price_efficiency: Hurst adjusted by entropy (more entropy = less efficient) ─
        max_ent  = np.log2(10.0)                          # max bits for 10 bins
        safe_ent = np.where(max_ent > 0, max_ent, 1.0)
        norm_ent = np.clip(entropy / safe_ent, 0.0, 1.0)
        price_efficiency = hurst * (1.0 - norm_ent * 0.5)

        # ── regime_consistency: vol regime aligns with trend persistence ─────
        regime_consistency = vol_regime * trend_persist

        out = pd.DataFrame(index=df.index)
        out["return_vol_ratio"]    = return_vol_ratio
        out["trend_quality"]       = trend_quality
        out["noise_ratio"]         = noise_ratio
        out["price_efficiency"]    = price_efficiency
        out["regime_consistency"]  = regime_consistency
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Cross-module composites: return_vol_ratio (normalised 5-bar return), "
                "trend_quality (ER × momentum persistence), noise_ratio (market noise "
                "× vol expansion), price_efficiency (Hurst × entropy adjustment), "
                "regime_consistency (vol regime × trend persistence). 5 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _STAT_ENGINE_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "low",
            tags       = ["composite", "cross_module", "statistics"],
        )
