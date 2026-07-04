"""Fair Value Gap (FVG) detection and tracking.

Extracted from the LuxAlgo Smart Money Concepts indicator logic.
Self-contained — depends only on base OHLCV columns.

Definition
----------
A Fair Value Gap is a three-candle pattern where the impulse candle
creates a price gap that leaves an unfilled region on the chart.

    Bullish FVG at bar i:
        high[i-2] < low[i]
        The gap zone is [high[i-2], low[i]].
        Price gapped UP — the lower region between high[i-2] and low[i]
        was never traded; it may act as future support.

    Bearish FVG at bar i:
        low[i-2] > high[i]
        The gap zone is [high[i], low[i-2]].
        Price gapped DOWN — the upper region between high[i] and low[i-2]
        was never traded; it may act as future resistance.

Minimum gap threshold
---------------------
A configurable threshold (default 0.0 pips = no filter) removes micro-gaps
caused by spread or rounding.  The threshold is expressed as a fraction of
close price (e.g. 0.0001 = 1 pip on a 4-decimal pair).

Mitigation
----------
A bullish FVG is mitigated when price closes BELOW the gap top (low[i]):
    close[j] <= fvg_bullish_top  for some j > i

A bearish FVG is mitigated when price closes ABOVE the gap bottom (high[i]):
    close[j] >= fvg_bearish_bottom  for some j > i

Only the MOST RECENT unmitigated FVG of each type is tracked.

No-repainting guarantee
-----------------------
FVG detection at bar i uses only high[i-2] and low[i] — fully causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class FairValueGapEngine(BaseFeature):
    """Detect and track the most recent unmitigated bullish and bearish FVG."""

    name:             str       = "fair_value_gaps"
    category:         str       = "market_structure"
    dependencies:     list[str] = []
    required_columns: list[str] = ["high", "low", "close"]

    # Minimum gap size as a fraction of close price (0 = no filter)
    _MIN_GAP_PCT: float = 0.0

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect FVG events and track the active gap zone through time.

        Returns
        -------
        pd.DataFrame
            10 float64 columns:
            fvg_bullish, fvg_bearish,
            fvg_bullish_top, fvg_bullish_bottom,
            fvg_bearish_top, fvg_bearish_bottom,
            fvg_bullish_active, fvg_bearish_active,
            fvg_bullish_age, fvg_bearish_age.
        """
        n      = len(df)
        high_  = df["high"].to_numpy()
        low_   = df["low"].to_numpy()
        close_ = df["close"].to_numpy()

        # ── Step 1: identify FVG bars (vectorised) ────────────────────────
        # Shift high and low by 2 to compare candle[i-2] vs candle[i]
        high2 = np.empty(n); high2[:] = np.nan
        low2  = np.empty(n); low2[:]  = np.nan
        high2[2:] = high_[:-2]
        low2[2:]  = low_[:-2]

        bull_gap  = low_ - high2    # positive = bullish gap
        bear_gap  = low2  - high_   # positive = bearish gap

        min_gap = close_ * self._MIN_GAP_PCT

        bull_fvg = (bull_gap > min_gap)
        bear_fvg = (bear_gap > min_gap)
        # Mask first 2 bars (no candle[i-2] available)
        bull_fvg[:2] = False
        bear_fvg[:2] = False

        # Gap top / bottom at each FVG bar
        bull_fvg_top    = np.where(bull_fvg, low_,   np.nan)   # low[i]
        bull_fvg_bottom = np.where(bull_fvg, high2,  np.nan)   # high[i-2]
        bear_fvg_top    = np.where(bear_fvg, low2,   np.nan)   # low[i-2]
        bear_fvg_bottom = np.where(bear_fvg, high_,  np.nan)   # high[i]

        # ── Step 2: sequential state machine for active-FVG tracking ──────
        a_bull_top = np.nan;  a_bull_bot = np.nan
        a_bear_top = np.nan;  a_bear_bot = np.nan
        a_bull_age = 0.0;     a_bear_age = 0.0

        bull_top_tr  = np.full(n, np.nan)
        bull_bot_tr  = np.full(n, np.nan)
        bear_top_tr  = np.full(n, np.nan)
        bear_bot_tr  = np.full(n, np.nan)
        bull_act_tr  = np.zeros(n)
        bear_act_tr  = np.zeros(n)
        bull_age_tr  = np.zeros(n)
        bear_age_tr  = np.zeros(n)

        for i in range(n):
            # Mitigation check (on current close vs active gap zone)
            if not np.isnan(a_bull_top):
                if close_[i] <= a_bull_top:   # close at or below the gap top
                    a_bull_top = np.nan;  a_bull_bot = np.nan;  a_bull_age = 0.0
                else:
                    a_bull_age += 1.0
            if not np.isnan(a_bear_top):
                if close_[i] >= a_bear_bot:   # close at or above the gap bottom
                    a_bear_top = np.nan;  a_bear_bot = np.nan;  a_bear_age = 0.0
                else:
                    a_bear_age += 1.0

            # New FVG at this bar supersedes the previous active one
            if bull_fvg[i]:
                a_bull_top = bull_fvg_top[i];  a_bull_bot = bull_fvg_bottom[i];  a_bull_age = 0.0
            if bear_fvg[i]:
                a_bear_top = bear_fvg_top[i];  a_bear_bot = bear_fvg_bottom[i];  a_bear_age = 0.0

            bull_top_tr[i] = a_bull_top;  bull_bot_tr[i] = a_bull_bot
            bear_top_tr[i] = a_bear_top;  bear_bot_tr[i] = a_bear_bot
            bull_act_tr[i] = 0.0 if np.isnan(a_bull_top) else 1.0
            bear_act_tr[i] = 0.0 if np.isnan(a_bear_top) else 1.0
            bull_age_tr[i] = a_bull_age
            bear_age_tr[i] = a_bear_age

        # ── Step 3: assemble output ────────────────────────────────────────
        out = pd.DataFrame(index=df.index)
        out["fvg_bullish"]        = bull_fvg.astype(float)
        out["fvg_bearish"]        = bear_fvg.astype(float)
        out["fvg_bullish_top"]    = bull_top_tr
        out["fvg_bullish_bottom"] = bull_bot_tr
        out["fvg_bearish_top"]    = bear_top_tr
        out["fvg_bearish_bottom"] = bear_bot_tr
        out["fvg_bullish_active"] = bull_act_tr
        out["fvg_bearish_active"] = bear_act_tr
        out["fvg_bullish_age"]    = bull_age_tr
        out["fvg_bearish_age"]    = bear_age_tr
        return out

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "LuxAlgo-style Fair Value Gap (FVG) detection.  Identifies "
                "three-candle price-gap patterns at both directions and tracks "
                "the most recent unmitigated gap zone (top, bottom, active flag, "
                "and age) through time.  Mitigated when close fills the gap."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = [
                "fvg_bullish", "fvg_bearish",
                "fvg_bullish_top", "fvg_bullish_bottom",
                "fvg_bearish_top", "fvg_bearish_bottom",
                "fvg_bullish_active", "fvg_bearish_active",
                "fvg_bullish_age", "fvg_bearish_age",
            ],
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = [
                "ICT", "smart_money", "market_structure",
                "FVG", "fair_value_gap", "imbalance",
            ],
        )
