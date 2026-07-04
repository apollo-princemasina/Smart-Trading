"""Equal Highs (EQH) and Equal Lows (EQL) detection.

Extracted from the LuxAlgo Smart Money Concepts indicator logic.
Depends on MarketStructureEngine for confirmed pivot levels.

Definition
----------
Equal Highs (EQH): Two consecutive confirmed minor pivot highs whose prices
    are within a configurable threshold of each other.  Markets create EQH
    when buy-side liquidity (stop orders above the highs) accumulates —
    price is likely to sweep these levels before a directional move.

Equal Lows (EQL): Two consecutive confirmed minor pivot lows whose prices
    are within a configurable threshold of each other.  Sell-side liquidity
    pools above EQL levels are targets for stop hunts.

Threshold
---------
Default: 0.05 % of the higher price.  Tuned for EURUSD.
Higher threshold → more EQH/EQL detected, lower precision.
Lower threshold  → fewer, more exact equal-level matches.

Output
------
eqh            — 1.0 at the bar of the SECOND equal high
eql            — 1.0 at the bar of the SECOND equal low
eqh_price      — price of the most recent EQH level (forward-filled)
eql_price      — price of the most recent EQL level (forward-filled)
eqh_age        — bars since most recent EQH (0 at EQH bar, increments after)
eql_age        — bars since most recent EQL
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry


@FeatureRegistry.register
class EqualHighsLowsEngine(BaseFeature):
    """Detect equal high and equal low liquidity pools from confirmed pivots."""

    name:             str       = "equal_highs_lows"
    category:         str       = "liquidity"
    dependencies:     list[str] = ["market_structure"]
    required_columns: list[str] = [
        "high", "low",
        "pivot_high",   # from market_structure — minor pivot flag
        "pivot_low",    # from market_structure — minor pivot flag
    ]

    # Threshold: max allowed difference as a fraction of price
    _THRESHOLD: float = 0.0005   # 0.05 % ≈ 5 pips on EURUSD

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare consecutive confirmed pivots to identify equal levels.

        Parameters
        ----------
        df:
            Enriched pipeline DataFrame with market_structure columns.

        Returns
        -------
        pd.DataFrame
            6 float64 columns: eqh, eql, eqh_price, eql_price,
            eqh_age, eql_age.
        """
        pivot_high = df["pivot_high"].astype(bool)
        pivot_low  = df["pivot_low"].astype(bool)

        out = pd.DataFrame(index=df.index)

        out["eqh"], out["eqh_price"] = self._detect_equal(
            df["high"], pivot_high, direction="high"
        )
        out["eql"], out["eql_price"] = self._detect_equal(
            df["low"], pivot_low, direction="low"
        )

        out["eqh_age"] = self._bars_since(out["eqh"].astype(bool))
        out["eql_age"] = self._bars_since(out["eql"].astype(bool))

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _detect_equal(
        cls,
        price:     pd.Series,
        pivots:    pd.Series,
        direction: str,         # "high" or "low"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Mark bars where two consecutive pivots of the same type are nearly equal.

        Returns
        -------
        equal_flags : pd.Series
            1.0 at the SECOND pivot bar in an equal-level pair.
        level_price : pd.Series
            Price of the most recent EQH/EQL, forward-filled to all bars.
        """
        equal_flags = pd.Series(0.0, index=price.index)
        level_price = pd.Series(np.nan, index=price.index)

        # Extract prices at confirmed pivot bars only
        pivot_prices = price.where(pivots, other=np.nan).dropna()
        if len(pivot_prices) < 2:
            return equal_flags, level_price.ffill()

        prev = pivot_prices.shift(1)
        ref  = pivot_prices if direction == "high" else prev

        # Relative difference between consecutive pivots
        rel_diff = (pivot_prices - prev).abs() / ref.abs().replace(0, np.nan)
        eq_mask  = rel_diff <= cls._THRESHOLD

        # Mark the second pivot in each equal pair
        eq_indices = pivot_prices.index[eq_mask.fillna(False).to_numpy()]
        equal_flags.loc[eq_indices] = 1.0

        # Store the EQH/EQL price (average of the two equal pivots)
        for idx in eq_indices:
            avg_price = (
                pivot_prices.loc[idx] + prev.loc[idx]
            ) / 2.0
            level_price.loc[idx] = avg_price

        return equal_flags, level_price.ffill()

    @staticmethod
    def _bars_since(event_flags: pd.Series) -> pd.Series:
        """Bars since the most recent True event (0 at the event bar)."""
        group = event_flags.astype(int).cumsum()
        return event_flags.groupby(group).cumcount().astype(float)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "LuxAlgo-style Equal Highs (EQH) and Equal Lows (EQL) "
                "detection.  Identifies liquidity pools where two consecutive "
                f"confirmed pivot highs or lows are within {self._THRESHOLD*100:.3f}% "
                "of each other.  EQH/EQL levels are prime stop-hunt targets "
                "and often precede significant directional moves."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = [
                "eqh", "eql",
                "eqh_price", "eql_price",
                "eqh_age", "eql_age",
            ],
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "low",
            tags       = [
                "ICT", "smart_money", "liquidity",
                "EQH", "EQL", "equal_highs", "equal_lows", "stop_hunt",
            ],
        )
