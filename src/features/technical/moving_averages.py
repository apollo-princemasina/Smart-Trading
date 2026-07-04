"""Moving average indicators: EMA, SMA, WMA, HMA plus derived slope/cross."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_PERIOD_WMA = 20
_PERIOD_HMA = 20
_SLOPE_LOOKBACK = 5   # bars used for ema_slope finite-difference

_MA_COLUMNS: list[str] = [
    "ema9", "ema20", "ema50", "ema100", "ema200",
    "sma20", "sma50", "sma100",
    "wma20", "hma20",
    "ema_slope", "ema_cross",
]


def _wma(arr: np.ndarray, period: int) -> np.ndarray:
    """Weighted moving average via fast numpy convolution (no Python loops).

    np.convolve reverses the kernel internally, so descending weights here
    produce ascending effective weights (newest bar gets highest weight).
    """
    w = np.arange(period, 0, -1, dtype=np.float64)   # [period, period-1, ..., 1]
    w /= w.sum()
    padded = np.concatenate([np.full(period - 1, arr[0]), arr.astype(np.float64)])
    return np.convolve(padded, w, mode="valid")


@FeatureRegistry.register
class MovingAveragesEngine(BaseFeature):
    """EMA (9/20/50/100/200), SMA (20/50/100), WMA(20), HMA(20) plus ema_slope and ema_cross."""

    name:             str       = "moving_averages"
    category:         str       = "technical"
    dependencies:     list[str] = []
    required_columns: list[str] = ["close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].to_numpy(dtype=np.float64)
        close_s = pd.Series(close, index=df.index)

        # ── EMAs ──────────────────────────────────────────────────────────────
        ema9   = close_s.ewm(span=9,   adjust=False).mean().to_numpy()
        ema20  = close_s.ewm(span=20,  adjust=False).mean().to_numpy()
        ema50  = close_s.ewm(span=50,  adjust=False).mean().to_numpy()
        ema100 = close_s.ewm(span=100, adjust=False).mean().to_numpy()
        ema200 = close_s.ewm(span=200, adjust=False).mean().to_numpy()

        # ── SMAs ──────────────────────────────────────────────────────────────
        sma20  = close_s.rolling(20,  min_periods=1).mean().to_numpy()
        sma50  = close_s.rolling(50,  min_periods=1).mean().to_numpy()
        sma100 = close_s.rolling(100, min_periods=1).mean().to_numpy()

        # ── WMA(20) and HMA(20) ────────────────────────────────────────────────
        wma20 = _wma(close, _PERIOD_WMA)

        half = _PERIOD_WMA // 2
        sqrt_n = max(2, int(np.sqrt(_PERIOD_WMA)))
        hma_src = 2.0 * _wma(close, half) - _wma(close, _PERIOD_WMA)
        hma20 = _wma(hma_src, sqrt_n)

        # ── Derived: slope and cross ───────────────────────────────────────────
        ema20_s = pd.Series(ema20, index=df.index)
        ema50_s = pd.Series(ema50, index=df.index)

        prev_ema20 = ema20_s.shift(_SLOPE_LOOKBACK).to_numpy()
        safe_prev  = np.where(prev_ema20 > 0, prev_ema20, 1.0)
        ema_slope  = np.where(prev_ema20 > 0,
                              (ema20 - prev_ema20) / safe_prev * 100.0,
                              0.0)
        ema_cross = np.sign(ema20 - ema50)

        out = pd.DataFrame(index=df.index)
        out["ema9"]       = ema9
        out["ema20"]      = ema20
        out["ema50"]      = ema50
        out["ema100"]     = ema100
        out["ema200"]     = ema200
        out["sma20"]      = sma20
        out["sma50"]      = sma50
        out["sma100"]     = sma100
        out["wma20"]      = wma20
        out["hma20"]      = hma20
        out["ema_slope"]  = ema_slope
        out["ema_cross"]  = ema_cross
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "EMA (9/20/50/100/200), SMA (20/50/100), WMA(20), HMA(20), "
                "plus ema_slope (5-bar pct change of EMA20) and "
                "ema_cross (sign of EMA20 − EMA50).  12 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _MA_COLUMNS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "low",
            tags       = ["moving_average", "trend", "ema", "sma", "wma", "hma"],
        )
