"""Distribution statistics: skewness, kurtosis, z-score, percentile rank, price rank."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_WINDOW = 20

_DIST_COLS: list[str] = [
    "skewness",
    "kurtosis",
    "zscore",
    "percentile_rank",
    "normalized_price",
    "price_rank",
]


def _skew(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mu = np.mean(x)
    s  = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 3))


def _kurt(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    mu = np.mean(x)
    s  = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 4) - 3.0)   # excess kurtosis


def _pct_rank(x: np.ndarray) -> float:
    """Percentile rank of the last element in x (0..100)."""
    return float(np.sum(x < x[-1]) / max(len(x) - 1, 1) * 100.0)


def _price_rank(x: np.ndarray) -> float:
    """Ordinal rank of the last element (0-based, normalised to [0,1])."""
    n = len(x)
    if n < 2:
        return 0.5
    return float(np.argsort(np.argsort(x))[-1] / max(n - 1, 1))


@FeatureRegistry.register
class DistributionEngine(BaseFeature):
    """Rolling skewness/kurtosis of log return; z-score, percentile rank, price rank of close."""

    name:             str       = "distribution"
    category:         str       = "statistics"
    dependencies:     list[str] = ["returns"]
    required_columns: list[str] = ["close", "log_return"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        close     = df["close"].to_numpy(dtype=np.float64)
        log_ret   = df["log_return"].to_numpy(dtype=np.float64)

        close_s  = pd.Series(close,   index=df.index)
        lr_s     = pd.Series(log_ret, index=df.index)

        # ── Rolling skewness / kurtosis of log_return ─────────────────────────
        skewness = lr_s.rolling(_WINDOW, min_periods=4).apply(_skew, raw=True).to_numpy()
        kurtosis = lr_s.rolling(_WINDOW, min_periods=4).apply(_kurt, raw=True).to_numpy()

        # Fill NaN in warm-up period with neutral values
        skewness = np.where(np.isnan(skewness), 0.0, skewness)
        kurtosis = np.where(np.isnan(kurtosis), 0.0, kurtosis)

        # ── Z-score of close relative to its rolling distribution ─────────────
        roll_mean = close_s.rolling(_WINDOW, min_periods=1).mean().to_numpy()
        roll_std  = close_s.rolling(_WINDOW, min_periods=1).std(ddof=0).to_numpy()
        safe_std  = np.where(roll_std > 0, roll_std, 1.0)
        zscore    = np.where(roll_std > 0, (close - roll_mean) / safe_std, 0.0)

        # ── Percentile rank (0–100) ────────────────────────────────────────────
        percentile_rank = close_s.rolling(_WINDOW, min_periods=2).apply(
            _pct_rank, raw=True).to_numpy()
        percentile_rank = np.where(np.isnan(percentile_rank), 50.0, percentile_rank)

        # ── Min-max normalised price in rolling window ────────────────────────
        roll_min = close_s.rolling(_WINDOW, min_periods=1).min().to_numpy()
        roll_max = close_s.rolling(_WINDOW, min_periods=1).max().to_numpy()
        rng      = roll_max - roll_min
        safe_rng = np.where(rng > 0, rng, 1.0)
        normalized_price = np.where(rng > 0, (close - roll_min) / safe_rng, 0.5)

        # ── Ordinal price rank within rolling window (0–1) ───────────────────
        price_rank = close_s.rolling(_WINDOW, min_periods=2).apply(
            _price_rank, raw=True).to_numpy()
        price_rank = np.where(np.isnan(price_rank), 0.5, price_rank)

        out = pd.DataFrame(index=df.index)
        out["skewness"]         = skewness
        out["kurtosis"]         = kurtosis
        out["zscore"]           = zscore
        out["percentile_rank"]  = percentile_rank
        out["normalized_price"] = normalized_price
        out["price_rank"]       = price_rank
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Rolling 20-bar distribution statistics: skewness and excess kurtosis "
                "of log_return; z-score, percentile rank (0–100), min-max normalised "
                "price, and ordinal price rank (0–1) of close. 6 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _DIST_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = ["distribution", "skewness", "kurtosis", "zscore",
                          "percentile", "statistics"],
        )
