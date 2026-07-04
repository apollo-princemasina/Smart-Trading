"""Market structure state tracker.

Tracks the key price levels that define the current structural context:
  last_major_high / last_major_low      — most recent confirmed major pivot
  last_internal_high / last_internal_low — most recent confirmed internal pivot

Distance features express the gap between current close and each reference
level as a signed percentage:
  Positive distance_to_last_major_high  → price is below the major high
  Negative distance_to_last_major_high  → price has broken above the major high
  Positive distance_to_last_major_low   → price is above the major low
  Negative distance_to_last_major_low   → price has broken below the major low

All four reference prices are forward-filled from the moment a pivot is
confirmed, so every bar carries valid structural context once enough pivots
have formed.  Bars before the first confirmed pivot carry NaN for reference
prices and 0.0 for distance features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StructureState:
    """Compute last-pivot reference prices and distance-to-structure features."""

    def compute(
        self,
        df: pd.DataFrame,
        pivot_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build structure state columns for every bar in df.

        Parameters
        ----------
        df:
            Base OHLCV DataFrame with at least ``close``.
        pivot_df:
            Output from PivotDetector.detect() — must contain
            ``major_pivot_high``, ``major_pivot_low``,
            ``_internal_pivot_high``, ``_internal_pivot_low``.

        Returns
        -------
        pd.DataFrame
            Eight float64 columns:
            ``last_major_high``, ``last_major_low``,
            ``last_internal_high``, ``last_internal_low``,
            ``distance_to_last_major_high``, ``distance_to_last_major_low``,
            ``distance_to_last_internal_high``, ``distance_to_last_internal_low``.
        """
        close = df["close"]
        out   = pd.DataFrame(index=df.index)

        # ── Reference prices: last confirmed pivot, forward-filled ────────
        out["last_major_high"] = (
            df["high"]
            .where(pivot_df["major_pivot_high"] == 1.0, other=np.nan)
            .ffill()
        )
        out["last_major_low"] = (
            df["low"]
            .where(pivot_df["major_pivot_low"] == 1.0, other=np.nan)
            .ffill()
        )
        out["last_internal_high"] = (
            df["high"]
            .where(pivot_df["_internal_pivot_high"] == 1.0, other=np.nan)
            .ffill()
        )
        out["last_internal_low"] = (
            df["low"]
            .where(pivot_df["_internal_pivot_low"] == 1.0, other=np.nan)
            .ffill()
        )

        # ── Signed percentage distances from close ────────────────────────
        # Positive → price has not yet reached the level (typical case).
        # Negative → price has moved through the level (structural break).
        safe_close = close.replace(0, np.nan)  # avoid division by zero on synthetic data

        out["distance_to_last_major_high"] = (
            (out["last_major_high"] - close) / safe_close * 100
        ).fillna(0.0)

        out["distance_to_last_major_low"] = (
            (close - out["last_major_low"]) / safe_close * 100
        ).fillna(0.0)

        out["distance_to_last_internal_high"] = (
            (out["last_internal_high"] - close) / safe_close * 100
        ).fillna(0.0)

        out["distance_to_last_internal_low"] = (
            (close - out["last_internal_low"]) / safe_close * 100
        ).fillna(0.0)

        return out
