"""Swing classification: Higher High / Lower High / Higher Low / Lower Low.

Takes confirmed pivot series from PivotDetector and classifies each pivot
by comparing it to the previous pivot of the same type (high vs high, low
vs low).

Also computes swing metadata useful for downstream ICT modules:
  swing_{high,low}_id        — sequential pivot counter (forward-filled)
  swing_{high,low}_price     — price of the most recent pivot (forward-filled)
  swing_{high,low}_duration  — bars since the last confirmed pivot of that type
  swing_{high,low}_range     — absolute price distance from the last opposite pivot
  swing_{high,low}_strength  — range normalised by 14-period ATR
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SwingAnalyzer:
    """Classify confirmed pivots as HH/LH or HL/LL and compute swing metadata."""

    def analyze(
        self,
        df: pd.DataFrame,
        pivot_highs: pd.Series,
        pivot_lows:  pd.Series,
    ) -> pd.DataFrame:
        """
        Classify pivot swing types and compute metadata.

        Parameters
        ----------
        df:
            Base OHLCV DataFrame (must have ``high``, ``low``, ``close``).
        pivot_highs:
            Boolean Series aligned to df.index — True at confirmed pivot highs.
        pivot_lows:
            Boolean Series aligned to df.index — True at confirmed pivot lows.

        Returns
        -------
        pd.DataFrame
            Columns: higher_high, lower_high, higher_low, lower_low,
                     swing_high_id, swing_low_id,
                     swing_high_price, swing_low_price,
                     swing_high_duration, swing_low_duration,
                     swing_high_range, swing_low_range,
                     swing_high_strength, swing_low_strength.
        """
        pivot_highs = pivot_highs.astype(bool)
        pivot_lows  = pivot_lows.astype(bool)

        # Prices only at pivot bars; NaN elsewhere.
        ph_prices = df["high"].where(pivot_highs, other=np.nan)
        pl_prices = df["low"].where(pivot_lows,   other=np.nan)

        out = pd.DataFrame(index=df.index)

        # ── Swing type classification ──────────────────────────────────────
        out["higher_high"], out["lower_high"] = self._classify(ph_prices, "high")
        out["higher_low"],  out["lower_low"]  = self._classify(pl_prices, "low")

        # ── Pivot IDs (1, 2, 3, … forward-filled) ─────────────────────────
        out["swing_high_id"] = self._swing_id(pivot_highs)
        out["swing_low_id"]  = self._swing_id(pivot_lows)

        # ── Last confirmed pivot price (forward-filled to every bar) ───────
        out["swing_high_price"] = ph_prices.ffill()
        out["swing_low_price"]  = pl_prices.ffill()

        # ── Bars since last confirmed pivot of each type ───────────────────
        out["swing_high_duration"] = self._bars_since(pivot_highs)
        out["swing_low_duration"]  = self._bars_since(pivot_lows)

        # ── Swing range: distance from last opposite pivot to this pivot ───
        out["swing_high_range"] = self._swing_range(
            pivot_prices=ph_prices,
            last_opposite=out["swing_low_price"],
        )
        out["swing_low_range"] = self._swing_range(
            pivot_prices=pl_prices,
            last_opposite=out["swing_high_price"],
        )

        # ── Swing strength: range / ATR ────────────────────────────────────
        atr = self._atr(df, period=14)
        out["swing_high_strength"] = (out["swing_high_range"] / atr.replace(0, np.nan)).fillna(0.0)
        out["swing_low_strength"]  = (out["swing_low_range"]  / atr.replace(0, np.nan)).fillna(0.0)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(
        pivot_prices: pd.Series,
        direction: str,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Compare each confirmed pivot to the previous one of the same type.

        For ``direction="high"``: returns (higher_high, lower_high).
        For ``direction="low"``:  returns (higher_low,  lower_low).

        Both returned series are float64 with 1.0 at the relevant pivot bar
        and 0.0 everywhere else.  The very first pivot is excluded (no
        previous pivot to compare against).
        """
        higher = pd.Series(0.0, index=pivot_prices.index)
        lower  = pd.Series(0.0, index=pivot_prices.index)

        pivots = pivot_prices.dropna()
        if len(pivots) < 2:
            return higher, lower

        prev = pivots.shift(1)
        has_prev = prev.notna()

        higher_mask = has_prev & (pivots > prev)
        lower_mask  = has_prev & (pivots < prev)

        higher.loc[pivots.index[higher_mask.to_numpy()]] = 1.0
        lower.loc[pivots.index[lower_mask.to_numpy()]]   = 1.0

        return higher, lower

    @staticmethod
    def _swing_id(pivot_flags: pd.Series) -> pd.Series:
        """Sequential integer ID for each confirmed pivot, forward-filled. NaN before first."""
        id_vals = pd.Series(np.nan, index=pivot_flags.index)
        pivot_idx = pivot_flags[pivot_flags].index
        id_vals.loc[pivot_idx] = np.arange(1, len(pivot_idx) + 1, dtype=float)
        return id_vals.ffill()

    @staticmethod
    def _bars_since(pivot_flags: pd.Series) -> pd.Series:
        """
        Number of bars elapsed since the most recent confirmed pivot.
        Returns 0 at the pivot bar itself, 1 at the next bar, 2 two bars later…

        Before the first pivot the counter starts from 0 at the first bar
        (treating the series start as the reference point).
        """
        is_pivot = pivot_flags.astype(bool)
        group = is_pivot.astype(int).cumsum()
        bars  = is_pivot.groupby(group).cumcount()
        return bars.astype(float)

    @staticmethod
    def _swing_range(
        pivot_prices:  pd.Series,
        last_opposite: pd.Series,
    ) -> pd.Series:
        """
        Absolute price distance from the last confirmed opposite-type pivot
        to this pivot.  Forward-filled to every subsequent bar, 0.0 if
        no reference is available yet.
        """
        range_at_pivot = (pivot_prices - last_opposite).abs()
        return range_at_pivot.ffill().fillna(0.0)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range (Wilder's smoothing approximated by SMA for simplicity)."""
        prev_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"]  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(window=period, min_periods=1).mean()
