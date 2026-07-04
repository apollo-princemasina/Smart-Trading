"""Multi-timeframe alignment — merges higher TF context onto the base (M15) frame.

No-lookahead guarantee
-----------------------
A higher-timeframe candle at time T covers the interval [T, T + duration).
That candle is only "complete" (safe to use as a feature) once the interval
has closed, i.e. at or after T + duration.

For each base candle at time `t`, we attach the latest higher-TF candle
whose completion time <= t.  This is implemented by shifting higher-TF
timestamps forward by one period before the merge_asof join, so only
fully-closed higher-TF candles are eligible.

Example (H1):
    H1 candle @ 09:00 closes at 10:00.  completion_time = 10:00.
    M15 candle @ 10:00: completion_time 10:00 <= 10:00  -> this H1 IS used.
    M15 candle @ 09:15: completion_time 10:00 > 09:15   -> NOT used (in-progress).
    The M15 @ 09:15 instead uses the H1 that closed at 09:00 (08:00 candle).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Candle duration used for the completion-time shift.
_DURATION: dict[str, pd.Timedelta] = {
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
    "W1": pd.Timedelta(weeks=1),
}

# Columns copied from each higher TF (spread/real_volume excluded for brevity)
_HTF_COLS = ["open", "high", "low", "close", "tick_volume"]


@dataclass
class MergeReport:
    base_tf:   str
    htf_count: int = 0
    base_rows: int = 0
    merged_rows: int = 0
    null_htf_rows: dict[str, int] = field(default_factory=dict)
    warnings:  list[str] = field(default_factory=list)


class TimeframeMerger:
    """
    Align multiple higher timeframes onto a base (lowest) timeframe.

    Parameters
    ----------
    base_tf : str
        The timeframe whose index is used (default "M15").
    higher_tfs : list[str]
        Higher timeframes to attach, e.g. ["H1", "H4", "D1", "W1"].
    """

    def __init__(
        self,
        base_tf:    str = "M15",
        higher_tfs: list[str] | None = None,
    ) -> None:
        self.base_tf    = base_tf
        self.higher_tfs = higher_tfs or ["H1", "H4", "D1", "W1"]

    def merge(
        self,
        base_df: pd.DataFrame,
        htf_dfs: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, MergeReport]:
        """
        Parameters
        ----------
        base_df : DataFrame
            The base-timeframe OHLCV (must have a UTC-aware `timestamp` col).
        htf_dfs : dict[str, DataFrame]
            Mapping from timeframe string to its cleaned OHLCV DataFrame.

        Returns
        -------
        merged : DataFrame
            Base rows with additional prefixed columns for each higher TF.
        report : MergeReport
        """
        report = MergeReport(base_tf=self.base_tf, base_rows=len(base_df))

        result = base_df.copy()
        result = self._ensure_utc(result, "timestamp")
        result = result.sort_values("timestamp").reset_index(drop=True)

        for tf in self.higher_tfs:
            if tf not in htf_dfs:
                report.warnings.append(f"{tf} not provided — skipping.")
                continue
            if tf not in _DURATION:
                report.warnings.append(f"No duration defined for {tf} — skipping.")
                continue

            htf = htf_dfs[tf].copy()
            htf = self._ensure_utc(htf, "timestamp")
            htf = htf.sort_values("timestamp").reset_index(drop=True)

            result = self._merge_one(result, htf, tf, report)
            report.htf_count += 1

        report.merged_rows = len(result)
        return result, report

    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_utc(df: pd.DataFrame, col: str) -> pd.DataFrame:
        ts = df[col]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            df[col] = pd.to_datetime(ts, utc=True)
        elif getattr(ts.dtype, "tz", None) is None:
            df[col] = ts.dt.tz_localize("UTC")
        return df

    def _merge_one(
        self,
        base: pd.DataFrame,
        htf:  pd.DataFrame,
        tf:   str,
        report: MergeReport,
    ) -> pd.DataFrame:
        duration = _DURATION[tf]

        # Shift timestamp forward by one candle duration to represent
        # when this candle is fully closed and safe to use as a feature.
        htf_shifted = htf[["timestamp"] + _HTF_COLS].copy()
        htf_shifted["_completion"] = htf_shifted["timestamp"] + duration

        # Rename feature columns to avoid clashing with base TF columns
        prefix = tf.lower()
        rename = {c: f"{prefix}_{c}" for c in _HTF_COLS}
        htf_shifted = htf_shifted.rename(columns=rename)

        # merge_asof: for each base row, find the latest htf row whose
        # completion time <= base timestamp (direction="backward").
        merged = pd.merge_asof(
            base.sort_values("timestamp"),
            htf_shifted.sort_values("_completion"),
            left_on="timestamp",
            right_on="_completion",
            direction="backward",
        )

        # Drop the helper completion column; keep the original htf timestamp
        # as a diagnostic column so downstream code can verify the join.
        merged = merged.drop(columns=["_completion"], errors="ignore")
        if "timestamp_y" in merged.columns:
            merged = merged.rename(columns={"timestamp_y": f"{prefix}_timestamp"})
        if "timestamp_x" in merged.columns:
            merged = merged.rename(columns={"timestamp_x": "timestamp"})

        # Count rows where no HTF candle was available (start of series)
        null_col = f"{prefix}_close"
        n_null = int(merged[null_col].isnull().sum()) if null_col in merged.columns else 0
        report.null_htf_rows[tf] = n_null
        if n_null:
            report.warnings.append(
                f"{n_null} base rows have no completed {tf} candle yet "
                f"(start of series before first {tf} candle closed)."
            )

        return merged.reset_index(drop=True)
