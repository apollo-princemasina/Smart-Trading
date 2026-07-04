"""Market microstructure: efficiency ratio, Hurst, fractal dimension, noise, smoothness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_ER_WINDOW       = 20    # Kaufman efficiency ratio
_HURST_WINDOW    = 40    # R/S Hurst exponent
_SMOOTH_WINDOW   = 20    # price smoothness
_AUTOCORR_WINDOW = 20    # lag-1 autocorrelation for mean-reversion / trend scores

_MICROSTRUCTURE_COLS: list[str] = [
    "efficiency_ratio",
    "hurst",
    "fractal_dimension",
    "market_noise",
    "directional_efficiency",
    "price_smoothness",
    "mean_reversion_score",
    "trend_score",
]


def _efficiency_ratio(x: np.ndarray) -> float:
    """Kaufman's Efficiency Ratio: |net move| / total path length ∈ [0, 1]."""
    if len(x) < 2:
        return 0.5
    net   = abs(x[-1] - x[0])
    path  = np.sum(np.abs(np.diff(x)))
    if path < 1e-10:
        return 0.0
    return float(np.clip(net / path, 0.0, 1.0))


def _rs_hurst(x: np.ndarray) -> float:
    """Single-scale R/S Hurst estimate from log-price increments.

    H > 0.5 → trending / persistent
    H ≈ 0.5 → random walk
    H < 0.5 → mean-reverting / anti-persistent
    """
    n = len(x)
    if n < 4:
        return 0.5
    mu  = np.mean(x)
    dev = np.cumsum(x - mu)
    r   = float(np.max(dev) - np.min(dev))
    s   = float(np.std(x, ddof=1))
    if s < 1e-10 or r < 1e-10:
        return 0.5
    return float(np.clip(np.log(r / s) / np.log(n), 0.0, 1.0))


def _autocorr_lag1(x: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation."""
    if len(x) < 3:
        return 0.0
    x1, x2 = x[:-1], x[1:]
    s1, s2  = np.std(x1, ddof=1), np.std(x2, ddof=1)
    if s1 < 1e-10 or s2 < 1e-10:
        return 0.0
    return float(np.mean((x1 - x1.mean()) * (x2 - x2.mean())) / (s1 * s2))


@FeatureRegistry.register
class MarketMicrostructureEngine(BaseFeature):
    """Efficiency ratio, Hurst exponent, fractal dimension, noise, smoothness, and regime scores."""

    name:             str       = "market_microstructure"
    category:         str       = "statistics"
    dependencies:     list[str] = ["returns"]
    required_columns: list[str] = ["close", "log_return"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        close   = df["close"].to_numpy(dtype=np.float64)
        log_ret = df["log_return"].to_numpy(dtype=np.float64)
        cls_s   = pd.Series(close,   index=df.index)
        lr_s    = pd.Series(log_ret, index=df.index)

        # ── Kaufman Efficiency Ratio ───────────────────────────────────────────
        efficiency_ratio = cls_s.rolling(_ER_WINDOW, min_periods=3).apply(
            _efficiency_ratio, raw=True).to_numpy()
        efficiency_ratio = np.where(np.isnan(efficiency_ratio), 0.5, efficiency_ratio)

        # ── Directional Efficiency (same as ER but on log_return path) ─────────
        directional_efficiency = lr_s.rolling(_ER_WINDOW, min_periods=3).apply(
            lambda x: _efficiency_ratio(np.cumsum(x)), raw=True).to_numpy()
        directional_efficiency = np.where(
            np.isnan(directional_efficiency), 0.5, directional_efficiency)

        # ── Hurst exponent via R/S on log returns ─────────────────────────────
        hurst = lr_s.rolling(_HURST_WINDOW, min_periods=8).apply(
            _rs_hurst, raw=True).to_numpy()
        hurst = np.where(np.isnan(hurst), 0.5, hurst)

        # ── Fractal dimension: FD = 2 − H ─────────────────────────────────────
        fractal_dimension = 2.0 - hurst

        # ── Market noise: complement of efficiency ratio ───────────────────────
        market_noise = 1.0 - efficiency_ratio

        # ── Price smoothness: 1 − σ(2nd diff) / σ(1st diff) ─────────────────
        first_d  = np.diff(log_ret, prepend=0.0)
        second_d = np.diff(first_d, prepend=0.0)
        fd_s     = pd.Series(first_d,  index=df.index)
        sd_s     = pd.Series(second_d, index=df.index)
        s1       = fd_s.rolling(_SMOOTH_WINDOW, min_periods=4).std(ddof=0).to_numpy()
        s2       = sd_s.rolling(_SMOOTH_WINDOW, min_periods=4).std(ddof=0).to_numpy()
        safe_s1  = np.where(s1 > 0, s1, 1.0)
        price_smoothness = np.where(s1 > 0,
                                    np.clip(1.0 - s2 / safe_s1, 0.0, 1.0), 0.5)
        price_smoothness = np.where(np.isnan(price_smoothness), 0.5, price_smoothness)

        # ── Lag-1 autocorrelation → mean-reversion and trend scores ──────────
        lag1 = lr_s.rolling(_AUTOCORR_WINDOW, min_periods=4).apply(
            _autocorr_lag1, raw=True).to_numpy()
        lag1 = np.where(np.isnan(lag1), 0.0, lag1)

        mean_reversion_score = np.maximum(0.0, -lag1)   # >0 when lag1<0
        trend_score          = np.maximum(0.0,  lag1)   # >0 when lag1>0

        out = pd.DataFrame(index=df.index)
        out["efficiency_ratio"]      = efficiency_ratio
        out["hurst"]                 = hurst
        out["fractal_dimension"]     = fractal_dimension
        out["market_noise"]          = market_noise
        out["directional_efficiency"]= directional_efficiency
        out["price_smoothness"]      = price_smoothness
        out["mean_reversion_score"]  = mean_reversion_score
        out["trend_score"]           = trend_score
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Kaufman Efficiency Ratio (20-bar), Hurst exponent R/S (40-bar), "
                "Fractal Dimension (2−H), Market Noise (1−ER), Directional Efficiency, "
                "Price Smoothness (1−σ₂/σ₁), Mean-Reversion Score (max(0,−ρ₁)), "
                "Trend Score (max(0,ρ₁)). 8 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _MICROSTRUCTURE_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "high",
            tags       = ["microstructure", "hurst", "efficiency", "fractal",
                          "noise", "statistics"],
        )
