"""Vectorized, no-repainting pivot detection for market structure analysis.

A pivot high at bar i is confirmed when the bar's high is:
  - strictly greater than all highs in bars [i-lookback, i-1]  (left side)
  - greater or equal to all highs in bars [i+1, i+lookback]    (right side)

Symmetrically for pivot lows (strictly lower left, less-equal right).

This is fully causal for offline feature generation: the right-side bars are
always historical data that already exists at the time of evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PivotConfig:
    """Lookback window sizes for each pivot strength tier."""
    major_lookback:    int = 15   # ~3.75 h on M15 — significant swing highs/lows
    minor_lookback:    int =  5   # ~1.25 h on M15 — standard structure
    internal_lookback: int =  3   # ~45 min on M15 — micro intraday structure


class PivotDetector:
    """
    Detect confirmed pivot highs and lows at three configurable strength tiers
    (major, minor, internal) using pure-pandas vectorised rolling operations.

    No Python-level loops over rows — the entire detection is implemented via
    pandas rolling (C/Cython under the hood) and numpy array slicing so that
    87 K rows run in well under a second.
    """

    def __init__(self, config: PivotConfig | None = None) -> None:
        self.config = config or PivotConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all pivot tiers and return a DataFrame of float64 binary columns.

        Parameters
        ----------
        df:
            Input OHLCV DataFrame with at least ``high`` and ``low`` columns.

        Returns
        -------
        pd.DataFrame
            Columns (float64, 1.0 = pivot confirmed, 0.0 = no pivot):
            ``pivot_high``, ``pivot_low``,
            ``major_pivot_high``, ``major_pivot_low``,
            ``minor_pivot_high``, ``minor_pivot_low``,
            ``_internal_pivot_high``, ``_internal_pivot_low``
            (internal columns prefixed with ``_`` are for downstream use only).
        """
        high = df["high"]
        low  = df["low"]

        minor_ph    = self._detect_high(high, self.config.minor_lookback)
        minor_pl    = self._detect_low(low,   self.config.minor_lookback)
        major_ph    = self._detect_high(high, self.config.major_lookback)
        major_pl    = self._detect_low(low,   self.config.major_lookback)
        internal_ph = self._detect_high(high, self.config.internal_lookback)
        internal_pl = self._detect_low(low,   self.config.internal_lookback)

        out = pd.DataFrame(index=df.index)
        out["pivot_high"]           = minor_ph.astype(float)
        out["pivot_low"]            = minor_pl.astype(float)
        out["major_pivot_high"]     = major_ph.astype(float)
        out["major_pivot_low"]      = major_pl.astype(float)
        out["minor_pivot_high"]     = minor_ph.astype(float)
        out["minor_pivot_low"]      = minor_pl.astype(float)
        out["_internal_pivot_high"] = internal_ph.astype(float)
        out["_internal_pivot_low"]  = internal_pl.astype(float)
        return out

    # ------------------------------------------------------------------
    # Internal helpers — fully vectorised
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_left_max(arr: np.ndarray, window: int) -> np.ndarray:
        """Max of arr[i-window : i] for each i (left-exclusive of i)."""
        # rolling(window).max() at position i includes i itself.
        # Shifting by 1 moves the window one step forward so position i
        # holds the max of [i-window, i-1].
        return (
            pd.Series(arr)
            .rolling(window=window, min_periods=window)
            .max()
            .shift(1)
            .to_numpy()
        )

    @staticmethod
    def _rolling_right_max(arr: np.ndarray, window: int) -> np.ndarray:
        """Max of arr[i+1 : i+window+1] for each i (right-exclusive of i)."""
        # Reverse the array, compute left-max on the reversed series, then
        # reverse the result back.  After reversal, position i in the reversed
        # series corresponds to position (n-1-i) in the original, so the
        # left-max over the reversed window gives the right-max in the original.
        rev = arr[::-1]
        rev_max = (
            pd.Series(rev)
            .rolling(window=window, min_periods=window)
            .max()
            .shift(1)
            .to_numpy()
        )
        return rev_max[::-1]

    @staticmethod
    def _rolling_left_min(arr: np.ndarray, window: int) -> np.ndarray:
        return (
            pd.Series(arr)
            .rolling(window=window, min_periods=window)
            .min()
            .shift(1)
            .to_numpy()
        )

    @staticmethod
    def _rolling_right_min(arr: np.ndarray, window: int) -> np.ndarray:
        rev = arr[::-1]
        rev_min = (
            pd.Series(rev)
            .rolling(window=window, min_periods=window)
            .min()
            .shift(1)
            .to_numpy()
        )
        return rev_min[::-1]

    @classmethod
    def _detect_high(cls, high: pd.Series, lookback: int) -> pd.Series:
        """
        Boolean Series — True at confirmed pivot high bars.

        Condition (no NaN comparison pitfalls — numpy returns False for NaN):
            high[i] > left_max[i]   (strictly greater than all left bars)
            high[i] >= right_max[i] (at least as high as all right bars)
        """
        arr       = high.to_numpy(dtype=float)
        left_max  = cls._rolling_left_max(arr, lookback)
        right_max = cls._rolling_right_max(arr, lookback)
        pivot     = (arr > left_max) & (arr >= right_max)
        return pd.Series(pivot, index=high.index)

    @classmethod
    def _detect_low(cls, low: pd.Series, lookback: int) -> pd.Series:
        """
        Boolean Series — True at confirmed pivot low bars.

        Condition:
            low[i] < left_min[i]   (strictly lower than all left bars)
            low[i] <= right_min[i] (at least as low as all right bars)
        """
        arr      = low.to_numpy(dtype=float)
        left_min = cls._rolling_left_min(arr, lookback)
        right_min = cls._rolling_right_min(arr, lookback)
        pivot    = (arr < left_min) & (arr <= right_min)
        return pd.Series(pivot, index=low.index)
