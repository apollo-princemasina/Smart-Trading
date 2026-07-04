"""Order Block (OB) detection.

Extracted from the LuxAlgo Smart Money Concepts indicator logic.
Depends on BosChochEngine output columns (bos_choch feature).

Definition
----------
An Order Block is the last opposing-direction candle immediately BEFORE a
BOS or CHoCH signal.  It represents an institutional order cluster — the
zone where smart money placed their orders before driving price through the
structural level.

    Bullish Order Block:
        The last bearish candle (close < open) before a bullish BOS/CHoCH.
        Zone: [min(open, close), max(open, close)] — the candle body.

    Bearish Order Block:
        The last bullish candle (close > open) before a bearish BOS/CHoCH.
        Zone: [min(open, close), max(open, close)] — the candle body.

Mitigation
----------
An OB is mitigated (invalidated) when price closes beyond the far edge of
the OB body — meaning institutional orders at that level have been absorbed:

    Bullish OB mitigated: close[i] < ob_bullish_bottom
    Bearish OB mitigated: close[i] > ob_bearish_top

Only the MOST RECENT unmitigated OB of each type is tracked.

No-repainting guarantee
-----------------------
OB identification uses only past OHLCV data and past BOS/CHoCH signals.
No future bar information is used in any calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

# Maximum bars to look back from a BOS bar when searching for the last
# opposing candle.  50 bars (≈12.5 h on M15) is ample for any realistic OB.
_OB_SEARCH_WINDOW: int = 50


@FeatureRegistry.register
class OrderBlockEngine(BaseFeature):
    """Detect and track the most recent bullish and bearish Order Blocks.

    Reads BOS/CHoCH signals from the enriched pipeline DataFrame.
    """

    name:             str       = "order_blocks"
    category:         str       = "market_structure"
    dependencies:     list[str] = ["bos_choch"]
    required_columns: list[str] = [
        "open", "high", "low", "close",
        "bos_bullish", "bos_bearish",
        "choch_bullish", "choch_bearish",
    ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify order blocks and track their state through time.

        Parameters
        ----------
        df:
            Enriched pipeline DataFrame containing OHLCV columns plus all
            bos_choch output columns.

        Returns
        -------
        pd.DataFrame
            10 float64 columns:
            ob_bullish, ob_bearish,
            ob_bullish_top, ob_bullish_bottom,
            ob_bearish_top, ob_bearish_bottom,
            ob_bullish_active, ob_bearish_active,
            price_in_bullish_ob, price_in_bearish_ob.
        """
        n      = len(df)
        open_  = df["open"].to_numpy()
        high_  = df["high"].to_numpy()
        low_   = df["low"].to_numpy()
        close_ = df["close"].to_numpy()

        # Combined BOS/CHoCH signals at each structural tier
        bull_signal = (
            (df["bos_bullish"] == 1.0) | (df["choch_bullish"] == 1.0)
        ).to_numpy()
        bear_signal = (
            (df["bos_bearish"] == 1.0) | (df["choch_bearish"] == 1.0)
        ).to_numpy()

        # ── Step 1: find the OB candle for each BOS/CHoCH bar ─────────────
        # ob_X_top/bottom are NaN everywhere except the bar that becomes an OB.
        ob_bull_top    = np.full(n, np.nan)
        ob_bull_bottom = np.full(n, np.nan)
        ob_bear_top    = np.full(n, np.nan)
        ob_bear_bottom = np.full(n, np.nan)
        ob_bull_flag   = np.zeros(n)
        ob_bear_flag   = np.zeros(n)

        bull_bos_idx = np.where(bull_signal)[0]
        bear_bos_idx = np.where(bear_signal)[0]

        for bos_i in bull_bos_idx:
            # Look back for the last bearish candle (close < open)
            start = max(0, bos_i - _OB_SEARCH_WINDOW)
            for j in range(bos_i - 1, start - 1, -1):
                if close_[j] < open_[j]:          # bearish candle
                    ob_bull_top[j]    = max(open_[j], close_[j])
                    ob_bull_bottom[j] = min(open_[j], close_[j])
                    ob_bull_flag[j]   = 1.0
                    break

        for bos_i in bear_bos_idx:
            # Look back for the last bullish candle (close > open)
            start = max(0, bos_i - _OB_SEARCH_WINDOW)
            for j in range(bos_i - 1, start - 1, -1):
                if close_[j] > open_[j]:          # bullish candle
                    ob_bear_top[j]    = max(open_[j], close_[j])
                    ob_bear_bottom[j] = min(open_[j], close_[j])
                    ob_bear_flag[j]   = 1.0
                    break

        # ── Step 2: track the most recent active OB through time ──────────
        active_bull_top    = np.nan
        active_bull_bottom = np.nan
        active_bear_top    = np.nan
        active_bear_bottom = np.nan

        bull_top_track    = np.full(n, np.nan)
        bull_bottom_track = np.full(n, np.nan)
        bear_top_track    = np.full(n, np.nan)
        bear_bottom_track = np.full(n, np.nan)
        bull_active_track = np.zeros(n)
        bear_active_track = np.zeros(n)

        for i in range(n):
            # Check bullish OB mitigation first
            if not np.isnan(active_bull_bottom):
                if close_[i] < active_bull_bottom:   # close below OB body
                    active_bull_top    = np.nan
                    active_bull_bottom = np.nan

            # Check bearish OB mitigation first
            if not np.isnan(active_bear_top):
                if close_[i] > active_bear_top:      # close above OB body
                    active_bear_top    = np.nan
                    active_bear_bottom = np.nan

            # Update with new OB formed at this bar (new OB supersedes old)
            if not np.isnan(ob_bull_top[i]):
                active_bull_top    = ob_bull_top[i]
                active_bull_bottom = ob_bull_bottom[i]
            if not np.isnan(ob_bear_top[i]):
                active_bear_top    = ob_bear_top[i]
                active_bear_bottom = ob_bear_bottom[i]

            bull_top_track[i]    = active_bull_top
            bull_bottom_track[i] = active_bull_bottom
            bear_top_track[i]    = active_bear_top
            bear_bottom_track[i] = active_bear_bottom
            bull_active_track[i] = 0.0 if np.isnan(active_bull_top) else 1.0
            bear_active_track[i] = 0.0 if np.isnan(active_bear_top) else 1.0

        # ── Step 3: assemble output DataFrame ─────────────────────────────
        out = pd.DataFrame(index=df.index)
        out["ob_bullish"]        = ob_bull_flag
        out["ob_bearish"]        = ob_bear_flag
        out["ob_bullish_top"]    = bull_top_track
        out["ob_bullish_bottom"] = bull_bottom_track
        out["ob_bearish_top"]    = bear_top_track
        out["ob_bearish_bottom"] = bear_bottom_track
        out["ob_bullish_active"] = bull_active_track
        out["ob_bearish_active"] = bear_active_track

        # Price inside OB zone
        out["price_in_bullish_ob"] = np.where(
            bull_active_track == 1.0,
            ((close_ >= bull_bottom_track) & (close_ <= bull_top_track)).astype(float),
            0.0,
        )
        out["price_in_bearish_ob"] = np.where(
            bear_active_track == 1.0,
            ((close_ >= bear_bottom_track) & (close_ <= bear_top_track)).astype(float),
            0.0,
        )

        return out

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "LuxAlgo-style Order Block detection.  Identifies the last "
                "opposing candle before each BOS/CHoCH and tracks the OB zone "
                "as an active support/resistance region until price mitigates "
                "it by closing beyond the far edge of the OB body."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = [
                "ob_bullish", "ob_bearish",
                "ob_bullish_top", "ob_bullish_bottom",
                "ob_bearish_top", "ob_bearish_bottom",
                "ob_bullish_active", "ob_bearish_active",
                "price_in_bullish_ob", "price_in_bearish_ob",
            ],
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = [
                "ICT", "smart_money", "market_structure",
                "order_block", "OB", "supply_demand",
            ],
        )
