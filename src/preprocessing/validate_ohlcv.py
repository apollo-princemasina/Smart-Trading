"""OHLCV data validator — 10 structural checks, no data modification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Expected candle frequency per timeframe string
_FREQ_MAP: dict[str, pd.Timedelta] = {
    "M15": pd.Timedelta(minutes=15),
    "H1":  pd.Timedelta(hours=1),
    "H4":  pd.Timedelta(hours=4),
    "D1":  pd.Timedelta(days=1),
    "W1":  pd.Timedelta(weeks=1),
}

# Gap threshold: a gap larger than this multiple of the expected interval
# is flagged as unexpected (ignores weekend/holiday windows).
_GAP_MULTIPLIER = 3

# Point-unit spread ceiling for EURUSD-class instruments.
# MT5 spread is in points (0.00001 per point for 5-decimal pairs).
# 150 points = 15 pips; very abnormal for liquid majors.
_SPREAD_WARNING_THRESHOLD = 150


@dataclass
class ValidationResult:
    """Summary of all validation checks for one timeframe DataFrame."""

    timeframe: str
    total_rows: int

    # Check counts
    duplicate_timestamps:   int = 0
    unsorted_timestamps:    bool = False
    missing_value_cells:    int = 0
    timezone_issues:        int = 0
    ohlc_violations:        int = 0
    negative_prices:        int = 0
    negative_volumes:       int = 0
    large_spreads:          int = 0
    constant_candles:       int = 0
    unexpected_gaps:        int = 0

    issues:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Hard failures that would break downstream feature engineering."""
        return len(self.issues) == 0


class OHLCVValidator:
    """
    Run 10 structural integrity checks on a raw OHLCV DataFrame.

    Does NOT modify the DataFrame. All findings are returned in a
    ValidationResult that downstream code can inspect or log.
    """

    def validate(self, df: pd.DataFrame, timeframe: str) -> ValidationResult:
        result = ValidationResult(
            timeframe=timeframe,
            total_rows=len(df),
        )

        if df.empty:
            result.issues.append("DataFrame is empty.")
            return result

        self._check_required_columns(df, result)
        if result.issues:
            # Column checks must pass before any other check can run safely
            return result

        self._check_duplicates(df, result)
        self._check_sorting(df, result)
        self._check_timezone(df, result)
        self._check_missing_values(df, result)
        self._check_ohlc_integrity(df, result)
        self._check_negative_prices(df, result)
        self._check_negative_volumes(df, result)
        self._check_extreme_spreads(df, result)
        self._check_constant_candles(df, result)
        self._check_unexpected_gaps(df, timeframe, result)

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required_columns(df: pd.DataFrame, result: ValidationResult) -> None:
        required = {"timestamp", "open", "high", "low", "close", "tick_volume"}
        missing = required - set(df.columns)
        if missing:
            result.issues.append(f"Missing required columns: {sorted(missing)}")

    @staticmethod
    def _check_duplicates(df: pd.DataFrame, result: ValidationResult) -> None:
        n = int(df["timestamp"].duplicated().sum())
        result.duplicate_timestamps = n
        if n:
            result.issues.append(f"{n} duplicate timestamps found.")

    @staticmethod
    def _check_sorting(df: pd.DataFrame, result: ValidationResult) -> None:
        if not df["timestamp"].is_monotonic_increasing:
            result.unsorted_timestamps = True
            result.issues.append("Timestamps are not sorted in ascending order.")

    @staticmethod
    def _check_timezone(df: pd.DataFrame, result: ValidationResult) -> None:
        ts = df["timestamp"]
        if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
            return  # timezone-aware — OK
        # Naive timestamps lose UTC reference during merge
        result.timezone_issues = len(df)
        result.warnings.append(
            f"timestamp column has no timezone info ({len(df)} rows). "
            "Expected UTC-aware dtype."
        )

    @staticmethod
    def _check_missing_values(df: pd.DataFrame, result: ValidationResult) -> None:
        price_cols = ["open", "high", "low", "close"]
        n = int(df[price_cols].isnull().sum().sum())
        result.missing_value_cells = n
        if n:
            result.issues.append(f"{n} NaN cells in price columns.")

    @staticmethod
    def _check_ohlc_integrity(df: pd.DataFrame, result: ValidationResult) -> None:
        bad = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"]  > df["open"]) |
            (df["low"]  > df["close"])
        )
        n = int(bad.sum())
        result.ohlc_violations = n
        if n:
            result.issues.append(f"{n} candles with invalid OHLC relationships.")

    @staticmethod
    def _check_negative_prices(df: pd.DataFrame, result: ValidationResult) -> None:
        bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        n = int(bad.sum())
        result.negative_prices = n
        if n:
            result.issues.append(f"{n} candles with zero or negative price values.")

    @staticmethod
    def _check_negative_volumes(df: pd.DataFrame, result: ValidationResult) -> None:
        if "tick_volume" not in df.columns:
            return
        n = int((df["tick_volume"] < 0).sum())
        result.negative_volumes = n
        if n:
            result.warnings.append(f"{n} candles with negative tick_volume.")

    @staticmethod
    def _check_extreme_spreads(df: pd.DataFrame, result: ValidationResult) -> None:
        if "spread" not in df.columns:
            return
        n = int((df["spread"] > _SPREAD_WARNING_THRESHOLD).sum())
        result.large_spreads = n
        if n:
            result.warnings.append(
                f"{n} candles with spread > {_SPREAD_WARNING_THRESHOLD} points "
                "(possible illiquid session or bad tick)."
            )

    @staticmethod
    def _check_constant_candles(df: pd.DataFrame, result: ValidationResult) -> None:
        same = (
            (df["open"] == df["high"]) &
            (df["high"] == df["low"]) &
            (df["low"]  == df["close"])
        )
        n = int(same.sum())
        result.constant_candles = n
        if n:
            result.warnings.append(
                f"{n} constant candles (O=H=L=C). "
                "May be synthetic filler bars."
            )

    @staticmethod
    def _check_unexpected_gaps(
        df: pd.DataFrame,
        timeframe: str,
        result: ValidationResult,
    ) -> None:
        if timeframe not in _FREQ_MAP or len(df) < 2:
            return

        expected = _FREQ_MAP[timeframe]
        threshold = expected * _GAP_MULTIPLIER

        diffs = df["timestamp"].sort_values().diff().dropna()

        # Gaps over threshold AND not a weekend boundary (Fri→Mon ≈ 60 h max)
        weekend_max = pd.Timedelta(hours=72)
        unexpected = diffs[(diffs > threshold) & (diffs > weekend_max)]
        n = len(unexpected)
        result.unexpected_gaps = n
        if n:
            result.warnings.append(
                f"{n} unexpected time gaps > {threshold} "
                "(excluding weekend windows)."
            )
