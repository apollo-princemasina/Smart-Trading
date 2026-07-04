"""OHLCV candle data validation for MT5-sourced datasets.

All MT5 downloads pass through validate_dataframe() before being saved
to disk. The validator checks schema, integrity, and temporal consistency,
and returns a structured ValidationReport so callers can decide how to
handle failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ── Schema ────────────────────────────────────────────────────────────────────

# Required columns produced by MT5Downloader.download()
REQUIRED_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]

# Expected candle interval per timeframe (used for gap detection).
# A gap is flagged when consecutive timestamps differ by more than
# GAP_MULTIPLIER × the expected interval (accounts for weekends/holidays).
FREQUENCY_MAP: dict[str, str] = {
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
    "W1":  "7D",
}

# Gaps up to this many times the expected interval are treated as normal
# market closures (weekends, bank holidays, MT5 server downtime).
GAP_MULTIPLIER = 4


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Structured result of a validate_dataframe() call."""
    passed:               bool
    missing_columns:      list[str]      = field(default_factory=list)
    duplicate_timestamps: int            = 0
    unsorted_timestamps:  bool           = False
    missing_values:       int            = 0
    invalid_ohlc:         int            = 0
    timezone_issues:      int            = 0
    gap_count:            int            = 0
    messages:             list[str]      = field(default_factory=list)


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_dataframe(
    df:        pd.DataFrame,
    timeframe: Optional[str] = None,
) -> ValidationReport:
    """
    Validate an MT5 OHLCV DataFrame for schema correctness and data integrity.

    The DataFrame must have a DatetimeIndex or a 'timestamp' column.
    When timeframe is supplied, gap detection is enabled.

    Checks performed:
        1. Required columns present.
        2. Index is a DatetimeIndex (or a 'timestamp' column exists).
        3. No duplicate timestamps.
        4. Timestamps sorted in ascending order.
        5. No NaN / missing values in OHLC + volume columns.
        6. Valid OHLC relationships (low ≤ open/close ≤ high).
        7. Index is UTC-aware.
        8. No unexpected gaps (optional, requires timeframe).

    Args:
        df:        DataFrame to validate.  Either the index or a 'timestamp'
                   column must carry datetime information.
        timeframe: Optional timeframe string (W1 | D1 | H4 | H1 | M15).
                   Enables gap detection when supplied.

    Returns:
        ValidationReport with pass/fail flag and detailed messages.
    """
    messages:        list[str] = []
    missing_columns: list[str] = []

    # 1 ── Schema check ────────────────────────────────────────────────────────
    present = set(df.columns) | (
        {str(df.index.name)} if df.index.name else set()
    )
    # Accept 'timestamp' either as a column or as the index name
    for col in REQUIRED_COLUMNS:
        if col == "timestamp":
            continue   # handled below by index check
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        messages.append(f"Missing required columns: {missing_columns}")

    # 2 ── Index / timestamp check ─────────────────────────────────────────────
    has_dt_index = isinstance(df.index, pd.DatetimeIndex)
    has_ts_col   = "timestamp" in df.columns

    if not has_dt_index and not has_ts_col:
        messages.append(
            "No DatetimeIndex and no 'timestamp' column found. "
            "Set timestamp as index or keep it as a column."
        )
        return ValidationReport(
            passed=False,
            missing_columns=missing_columns,
            messages=messages,
        )

    # Normalise: work with a Series of timestamps for the remaining checks
    ts: pd.Series = df.index.to_series() if has_dt_index else df["timestamp"]

    # 3 ── Duplicate timestamps ────────────────────────────────────────────────
    duplicate_timestamps = int(ts.duplicated().sum())
    if duplicate_timestamps:
        messages.append(f"{duplicate_timestamps} duplicate timestamp(s) found.")

    # 4 ── Sort order ──────────────────────────────────────────────────────────
    unsorted = not ts.is_monotonic_increasing
    if unsorted:
        messages.append("Timestamps are not in ascending order.")

    # 5 ── Missing values ──────────────────────────────────────────────────────
    value_cols    = [c for c in REQUIRED_COLUMNS if c != "timestamp" and c in df.columns]
    missing_vals  = int(df[value_cols].isna().sum().sum()) if value_cols else 0
    if missing_vals:
        messages.append(f"{missing_vals} missing value(s) in OHLCV/volume columns.")

    # 6 ── OHLC relationship integrity ─────────────────────────────────────────
    invalid_ohlc = 0
    ohlc_cols = {"open", "high", "low", "close"}
    if ohlc_cols.issubset(df.columns):
        bad = (
            (df["low"]  > df["open"])  |
            (df["low"]  > df["close"]) |
            (df["high"] < df["open"])  |
            (df["high"] < df["close"]) |
            (df["low"]  > df["high"])
        )
        invalid_ohlc = int(bad.sum())
        if invalid_ohlc:
            messages.append(
                f"{invalid_ohlc} row(s) with invalid OHLC relationships "
                "(e.g. low > high or high < open)."
            )

    # 7 ── Timezone awareness ──────────────────────────────────────────────────
    timezone_issues = 0
    idx_tz = df.index.tz if has_dt_index else getattr(ts.dt, "tz", None)
    if idx_tz is None or str(idx_tz).upper() not in ("UTC", "UTC+00:00"):
        timezone_issues = 1
        messages.append(
            "Timestamps must be UTC-aware (datetime64[ns, UTC]). "
            f"Found tz={idx_tz!r}."
        )

    # 8 ── Gap detection (optional) ────────────────────────────────────────────
    gap_count = 0
    if timeframe is not None and timeframe.upper() in FREQUENCY_MAP and len(ts) > 1:
        tf_key    = timeframe.upper()
        expected  = pd.Timedelta(FREQUENCY_MAP[tf_key])
        threshold = expected * GAP_MULTIPLIER
        deltas    = ts.diff().dropna()
        gap_count = int((deltas > threshold).sum())
        if gap_count:
            messages.append(
                f"{gap_count} gap(s) exceeding {threshold} detected "
                f"(threshold = {GAP_MULTIPLIER}× expected {expected})."
            )

    passed = not messages
    if passed:
        messages.append("Validation passed.")

    return ValidationReport(
        passed               = passed,
        missing_columns      = missing_columns,
        duplicate_timestamps = duplicate_timestamps,
        unsorted_timestamps  = unsorted,
        missing_values       = missing_vals,
        invalid_ohlc         = invalid_ohlc,
        timezone_issues      = timezone_issues,
        gap_count            = gap_count,
        messages             = messages,
    )
