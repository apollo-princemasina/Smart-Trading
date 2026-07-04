"""Per-timeframe alignment: merge HTF features onto an M15 DatetimeIndex."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .timeframe_mapper import TimeframeMapper

logger = logging.getLogger(__name__)

# Prefix for internal bar-open metadata columns (used by FusionValidator).
_BAR_OPEN_PREFIX = "__bar_open_"


class FeatureAligner:
    """
    Aligns a higher-timeframe feature DataFrame onto a base (M15) index.

    No-look-ahead guarantee
    -----------------------
    A bar's data is available only *after* that bar closes.  For a bar with
    open-time T and duration D:

        available_at = T + D

    We shift the HTF index by +1 period and use
    ``merge_asof(direction='backward')`` so each M15 bar at time T receives
    features from the latest HTF bar whose ``available_at ≤ T``.

    Example (H1 bars at 08:00, 09:00):
      - H1 08:00 → available_at = 09:00
      - M15 bars at 09:00, 09:15, 09:30, 09:45 → use H1 08:00  ✓
      - M15 bars at 10:00, 10:15, … → use H1 09:00             ✓
    """

    def align(
        self,
        htf_df: pd.DataFrame,
        tf: str,
        base_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Align *htf_df* features onto *base_index* (M15 timestamps).

        Returns a DataFrame indexed on *base_index* with every column
        prefixed ``{prefix}_`` plus an internal ``__bar_open_{tf}`` column
        that records the original bar open-time for look-ahead validation.
        """
        tf_norm = TimeframeMapper.validate(tf)
        prefix  = TimeframeMapper.prefix(tf_norm)
        offset  = TimeframeMapper.timedelta(tf_norm)

        if htf_df.empty:
            logger.warning("Empty HTF DataFrame for %s — NaN fill", tf_norm)
            cols = [f"{prefix}_{c}" for c in htf_df.columns]
            return pd.DataFrame(np.nan, index=base_index, columns=cols)

        # Dedup (keep last) and sort
        if htf_df.index.duplicated().any():
            logger.warning("Duplicate timestamps in %s data — keeping last", tf_norm)
            htf_df = htf_df[~htf_df.index.duplicated(keep="last")]
        htf_df = htf_df.sort_index()

        # Prefix feature columns
        htf_p = htf_df.rename(columns={c: f"{prefix}_{c}" for c in htf_df.columns})

        # Store original bar-open time for post-fusion look-ahead validation
        bar_open_col = f"{_BAR_OPEN_PREFIX}{tf_norm}"
        htf_p = htf_p.copy()
        htf_p[bar_open_col] = htf_df.index  # original open times (before shift)

        # Shift index: bar open → available_at (bar close = open + 1 period)
        htf_p.index = htf_df.index + offset
        htf_p.index.name = "_avail_at"

        # Reset so _avail_at becomes a plain column usable by merge_asof
        right = htf_p.reset_index()  # column "_avail_at" now exists

        # Left: M15 timestamps as a plain column
        left = pd.DataFrame({"_m15_ts": base_index})

        # For each M15 bar, find the latest HTF bar with available_at ≤ m15_ts
        merged = pd.merge_asof(
            left.sort_values("_m15_ts"),
            right.sort_values("_avail_at"),
            left_on   = "_m15_ts",
            right_on  = "_avail_at",
            direction = "backward",
        )

        merged.index = base_index
        merged = merged.drop(columns=["_m15_ts", "_avail_at"], errors="ignore")
        return merged

    def prefix_base(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Add ``m15_`` prefix to all base (M15) columns — no index shift."""
        return base_df.rename(columns={c: f"m15_{c}" for c in base_df.columns})

    @staticmethod
    def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove all ``__bar_open_*`` metadata columns from *df*."""
        internal = [c for c in df.columns if c.startswith(_BAR_OPEN_PREFIX)]
        return df.drop(columns=internal)

    @staticmethod
    def bar_open_columns(df: pd.DataFrame) -> dict[str, str]:
        """Return ``{tf_norm: column_name}`` for every internal bar-open column."""
        return {
            c[len(_BAR_OPEN_PREFIX):]: c
            for c in df.columns
            if c.startswith(_BAR_OPEN_PREFIX)
        }
