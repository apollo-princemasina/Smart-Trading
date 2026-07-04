"""Break of Structure (BOS) and Change of Character (CHoCH) detection.

Extracted from the LuxAlgo Smart Money Concepts indicator logic.
Depends on the MarketStructureEngine output columns (market_structure feature).

Definitions
-----------
BOS (Break of Structure)
    A break of a swing level in the SAME direction as the prevailing trend.
    Signals trend continuation.

CHoCH (Change of Character)
    A break of a swing level AGAINST the prevailing trend.
    Signals a potential trend reversal.

Two structural tiers
--------------------
Internal structure  (i-prefix) — uses the minor (5-bar) swing pivot levels.
    Fast-moving, generates more signals.  Used for entry precision.

Swing structure     — uses the major (15-bar) swing pivot levels.
    Slower, generates fewer but higher-conviction signals.
    Used for bias confirmation.

BOS / CHoCH classification logic
---------------------------------
    Bullish break (close crosses above structural level):
        prev_trend == +1  →  bos_bullish   (trend continuation)
        prev_trend != +1  →  choch_bullish (potential reversal to bullish)

    Bearish break (close crosses below structural level):
        prev_trend == -1  →  bos_bearish   (trend continuation)
        prev_trend != -1  →  choch_bearish (potential reversal to bearish)

Crossover detection
-------------------
A break is registered on the FIRST bar where close moves through the level.
Subsequent bars above/below the same level do not re-trigger the signal,
because the crossover condition requires the previous close to be on the
other side of the level.

No-repainting guarantee
-----------------------
All calculations use only past data (previous-bar reference levels via
.shift(1)) and the base OHLCV columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class BosChochEngine(BaseFeature):
    """BOS and CHoCH detection at internal and swing structural tiers.

    Reads market structure columns from the enriched pipeline DataFrame
    (produced by MarketStructureEngine).  Declares a dependency so the
    pipeline always runs MarketStructureEngine first.
    """

    name:             str       = "bos_choch"
    category:         str       = "market_structure"
    dependencies:     list[str] = ["market_structure"]
    required_columns: list[str] = [
        "close",
        "trend",             # from market_structure
        "swing_high_price",  # from market_structure — minor pivot high (ffill)
        "swing_low_price",   # from market_structure — minor pivot low (ffill)
        "last_major_high",   # from market_structure — major pivot high (ffill)
        "last_major_low",    # from market_structure — major pivot low (ffill)
    ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect BOS and CHoCH events at both structural tiers.

        Parameters
        ----------
        df:
            Enriched pipeline DataFrame — contains base OHLCV columns PLUS
            all columns from the ``market_structure`` feature generator.

        Returns
        -------
        pd.DataFrame
            10 float64 columns (all 0.0 or 1.0 except structure_bias and
            bars_since_structure_break):
            ibos_bullish, ibos_bearish, ichoch_bullish, ichoch_bearish,
            bos_bullish, bos_bearish, choch_bullish, choch_bearish,
            structure_bias, bars_since_structure_break.
        """
        close       = df["close"]
        prev_trend  = df["trend"].shift(1).fillna(0.0)

        out = pd.DataFrame(index=df.index)

        # ── Internal structure (minor/5-bar pivot levels) ─────────────────
        int_high = df["swing_high_price"]  # last minor pivot high, ffill'd
        int_low  = df["swing_low_price"]   # last minor pivot low,  ffill'd

        i_bull_break, i_bear_break = self._crossover(close, int_high, int_low)

        out["ibos_bullish"]   = (i_bull_break & (prev_trend ==  1.0)).astype(float)
        out["ibos_bearish"]   = (i_bear_break & (prev_trend == -1.0)).astype(float)
        out["ichoch_bullish"] = (i_bull_break & (prev_trend !=  1.0)).astype(float)
        out["ichoch_bearish"] = (i_bear_break & (prev_trend != -1.0)).astype(float)

        # ── Swing structure (major/15-bar pivot levels) ───────────────────
        sw_high = df["last_major_high"]    # last major pivot high, ffill'd
        sw_low  = df["last_major_low"]     # last major pivot low,  ffill'd

        s_bull_break, s_bear_break = self._crossover(close, sw_high, sw_low)

        out["bos_bullish"]   = (s_bull_break & (prev_trend ==  1.0)).astype(float)
        out["bos_bearish"]   = (s_bear_break & (prev_trend == -1.0)).astype(float)
        out["choch_bullish"] = (s_bull_break & (prev_trend !=  1.0)).astype(float)
        out["choch_bearish"] = (s_bear_break & (prev_trend != -1.0)).astype(float)

        # ── Forward-filled structure bias from last swing BOS/CHoCH ───────
        # Columns are float64 (0.0 / 1.0), so compare rather than bitwise-or.
        any_break = (
            (out["bos_bullish"] == 1.0) | (out["choch_bullish"] == 1.0) |
            (out["bos_bearish"] == 1.0) | (out["choch_bearish"] == 1.0)
        )
        raw_bias = pd.Series(np.nan, index=df.index)
        raw_bias[out["bos_bullish"] == 1.0]   = 1.0
        raw_bias[out["choch_bullish"] == 1.0] = 1.0
        raw_bias[out["bos_bearish"] == 1.0]   = -1.0
        raw_bias[out["choch_bearish"] == 1.0] = -1.0
        out["structure_bias"] = raw_bias.ffill().fillna(0.0)

        # ── Bars since any swing BOS/CHoCH ────────────────────────────────
        out["bars_since_structure_break"] = self._bars_since(any_break.astype(bool))

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crossover(
        close: pd.Series,
        ref_high: pd.Series,
        ref_low:  pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Detect close crossovers of the reference high and low levels.

        A bullish crossover fires on the first bar where:
            close[i] > ref_high[i-1]  AND  close[i-1] <= ref_high[i-1]

        A bearish crossover fires on the first bar where:
            close[i] < ref_low[i-1]  AND  close[i-1] >= ref_low[i-1]

        NaN reference levels produce False (no break) via numpy comparison
        semantics, correctly suppressing signals before the first pivot.
        """
        prev_high = ref_high.shift(1)
        prev_low  = ref_low.shift(1)

        bull = (close > prev_high) & (close.shift(1) <= prev_high)
        bear = (close < prev_low)  & (close.shift(1) >= prev_low)

        return bull.fillna(False), bear.fillna(False)

    @staticmethod
    def _bars_since(event_flags: pd.Series) -> pd.Series:
        """Bars elapsed since the most recent True bar (0 at the event bar)."""
        group     = event_flags.astype(int).cumsum()
        bars_since = event_flags.groupby(group).cumcount()
        return bars_since.astype(float)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "LuxAlgo-style BOS (Break of Structure) and CHoCH (Change of "
                "Character) detection at both internal (minor pivot) and swing "
                "(major pivot) structural levels.  BOS signals trend "
                "continuation; CHoCH signals a potential trend reversal."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = [
                "ibos_bullish", "ibos_bearish", "ichoch_bullish", "ichoch_bearish",
                "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
                "structure_bias", "bars_since_structure_break",
            ],
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = [
                "ICT", "smart_money", "market_structure",
                "BOS", "CHoCH", "break_of_structure", "change_of_character",
            ],
        )
