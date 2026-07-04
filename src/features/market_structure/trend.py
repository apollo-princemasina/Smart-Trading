"""Market trend engine — derives bias from the swing sequence.

Trend is determined by the agreement between the most recent swing high type
(HH or LH) and the most recent swing low type (HL or LL):

    HH + HL  →  Bullish   (+1.0)
    LH + LL  →  Bearish   (-1.0)
    Mixed    →  Neutral    (0.0)

Three output columns are produced:
  trend          — current market bias (+1 / -1 / 0)
  trend_duration — consecutive bars in the current bias state
  trend_strength — cumulative count of confirming swing events since the last
                   trend change (each HH or HL in a bullish period adds 1;
                   each LH or LL in a bearish period adds 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TrendEngine:
    """Derive market trend bias from a swing classification DataFrame."""

    def compute(
        self,
        df: pd.DataFrame,
        swing_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute trend, trend_duration, and trend_strength.

        Parameters
        ----------
        df:
            Base OHLCV DataFrame (index used for alignment).
        swing_df:
            Output from SwingAnalyzer.analyze() — must contain columns
            ``higher_high``, ``lower_high``, ``higher_low``, ``lower_low``.

        Returns
        -------
        pd.DataFrame
            Three float64 columns: ``trend``, ``trend_duration``,
            ``trend_strength``.
        """
        idx = df.index
        out = pd.DataFrame(index=idx)

        hh = swing_df["higher_high"]
        lh = swing_df["lower_high"]
        hl = swing_df["higher_low"]
        ll = swing_df["lower_low"]

        # ── Forward-fill last swing type for pivot highs (+1=HH, -1=LH) ──
        last_high_type = self._last_swing_type(idx, bull_event=hh, bear_event=lh)

        # ── Forward-fill last swing type for pivot lows (+1=HL, -1=LL) ───
        last_low_type = self._last_swing_type(idx, bull_event=hl, bear_event=ll)

        # ── Trend: both types must agree ──────────────────────────────────
        bullish = (last_high_type == 1.0) & (last_low_type == 1.0)
        bearish = (last_high_type == -1.0) & (last_low_type == -1.0)

        trend = pd.Series(0.0, index=idx)
        trend[bullish] = 1.0
        trend[bearish] = -1.0
        out["trend"] = trend

        # ── Trend duration: consecutive bars in the same trend state ──────
        trend_changed = (trend != trend.shift(1)).fillna(True)
        trend_group   = trend_changed.cumsum()

        out["trend_duration"] = (
            trend.groupby(trend_group).cumcount() + 1
        ).astype(float)

        # ── Trend strength: confirming swing events since last change ─────
        confirming = pd.Series(0.0, index=idx)
        confirming[bullish & ((hh == 1.0) | (hl == 1.0))] = 1.0
        confirming[bearish & ((lh == 1.0) | (ll == 1.0))] = 1.0

        out["trend_strength"] = (
            confirming.groupby(trend_group).cumsum()
        ).astype(float)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_swing_type(
        idx: pd.Index,
        bull_event: pd.Series,
        bear_event: pd.Series,
    ) -> pd.Series:
        """
        Forward-fill the most recent swing classification.

        Sets +1.0 at bull_event bars, -1.0 at bear_event bars, then
        forward-fills so every bar carries the last known bias.
        Bars before the first event are 0.0 (neutral).
        """
        raw = pd.Series(np.nan, index=idx)
        raw[bull_event == 1.0] = 1.0
        raw[bear_event == 1.0] = -1.0
        return raw.ffill().fillna(0.0)
