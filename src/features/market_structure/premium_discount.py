"""Premium and Discount zone classification.

Extracted from the LuxAlgo Smart Money Concepts indicator logic.
Depends on the MarketStructureEngine for major pivot reference levels.

Definition
----------
The Premium / Discount framework divides the range between the last confirmed
major swing high and the last confirmed major swing low into three zones:

    Premium (above 50%):  price > equilibrium → expensive, favour sells
    Equilibrium (≈50%):   price near the 50% level ± equilibrium_band
    Discount (below 50%): price < equilibrium → cheap, favour buys

Equilibrium band (default: 5% of range width on each side of 50%)
keeps the neutral zone from being too narrow on range-bound markets.

Output
------
pd_ratio               — (close - last_major_low) / range  ∈ [0, 1]
                          0 = at major low, 1 = at major high
pd_equilibrium         — price of the 50 % equilibrium level
pd_distance_from_eq    — signed % distance from equilibrium
                          positive = above eq (premium)
                          negative = below eq (discount)
pd_zone                — +1.0 premium, 0.0 equilibrium, −1.0 discount
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

# Fraction of range on EACH SIDE of 50% that counts as equilibrium.
# 0.05 means the equilibrium band covers [45 %, 55 %] of the range.
_EQ_BAND: float = 0.05


@FeatureRegistry.register
class PremiumDiscountEngine(BaseFeature):
    """Classify price position within the current major swing range."""

    name:             str       = "premium_discount"
    category:         str       = "market_structure"
    dependencies:     list[str] = ["market_structure"]
    required_columns: list[str] = [
        "close",
        "last_major_high",  # from market_structure
        "last_major_low",   # from market_structure
    ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute premium/discount zone metrics.

        Parameters
        ----------
        df:
            Enriched pipeline DataFrame with market_structure columns.

        Returns
        -------
        pd.DataFrame
            4 float64 columns: pd_ratio, pd_equilibrium,
            pd_distance_from_eq, pd_zone.
        """
        close    = df["close"]
        maj_high = df["last_major_high"]
        maj_low  = df["last_major_low"]

        rng = (maj_high - maj_low).replace(0, np.nan)   # avoid div-by-zero
        eq  = maj_low + rng * 0.5

        out = pd.DataFrame(index=df.index)

        out["pd_ratio"] = (
            ((close - maj_low) / rng)
            .clip(0.0, 1.0)
            .fillna(0.5)                    # default to equilibrium when no range
        )

        out["pd_equilibrium"] = eq

        out["pd_distance_from_eq"] = (
            (close - eq) / eq.replace(0, np.nan) * 100
        ).fillna(0.0)

        # Zone classification
        ratio = out["pd_ratio"]
        zone = pd.Series(0.0, index=df.index)
        zone[ratio > 0.5 + _EQ_BAND]  =  1.0    # premium
        zone[ratio < 0.5 - _EQ_BAND]  = -1.0    # discount
        out["pd_zone"] = zone

        return out

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "LuxAlgo-style Premium & Discount zone classification.  "
                "Divides the range between the last major swing high and low "
                "into Premium (above 55%), Equilibrium (45-55%), and Discount "
                "(below 45%) zones.  ICT traders use this to bias trade "
                "direction: buy from discount, sell from premium."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = [
                "pd_ratio", "pd_equilibrium",
                "pd_distance_from_eq", "pd_zone",
            ],
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "low",
            tags       = [
                "ICT", "smart_money", "market_structure",
                "premium", "discount", "equilibrium", "PD_array",
            ],
        )
