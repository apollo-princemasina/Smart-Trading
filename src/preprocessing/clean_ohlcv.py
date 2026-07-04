"""OHLCV data cleaner — fixes structural issues, never alters price values."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "tick_volume"]


@dataclass
class CleaningReport:
    """Records exactly what was removed or repaired."""

    rows_input:      int = 0
    rows_output:     int = 0
    duplicates_removed: int = 0
    invalid_ohlc_removed: int = 0
    zero_price_removed: int = 0
    rows_sorted:     bool = False
    tz_coerced:      bool = False
    actions:         list[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.rows_input - self.rows_output


class OHLCVCleaner:
    """
    Fix structural problems in a raw OHLCV DataFrame.

    Cleaning policy:
      - Remove exact duplicate timestamps (keep first occurrence).
      - Sort by timestamp ascending.
      - Drop candles where high < low (fundamentally corrupt price).
      - Drop candles with zero or negative prices.
      - Localize naive timestamps to UTC (non-destructive interpretation).
      - Cast OHLC columns to float64 and volume columns to int64.
      - Do NOT forward-fill or interpolate missing candles — gaps are
        information that the feature-engineering step uses.
    """

    def clean(
        self,
        df: pd.DataFrame,
        timeframe: str = "",
    ) -> tuple[pd.DataFrame, CleaningReport]:
        report = CleaningReport(rows_input=len(df))
        df = df.copy()

        df = self._coerce_types(df, report)
        df = self._ensure_utc(df, report)
        df = self._remove_duplicates(df, report)
        df = self._sort_timestamps(df, report)
        df = self._drop_invalid_ohlc(df, report)
        df = self._drop_zero_prices(df, report)
        df = df.reset_index(drop=True)

        report.rows_output = len(df)
        return df, report

    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_types(df: pd.DataFrame, _report: CleaningReport) -> pd.DataFrame:
        price_cols  = ["open", "high", "low", "close"]
        volume_cols = [c for c in ["tick_volume", "real_volume"] if c in df.columns]
        spread_cols = [c for c in ["spread"] if c in df.columns]

        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        for col in volume_cols + spread_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

        return df

    @staticmethod
    def _ensure_utc(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        ts = df["timestamp"]

        if pd.api.types.is_datetime64_any_dtype(ts):
            if getattr(ts.dtype, "tz", None) is None:
                df["timestamp"] = ts.dt.tz_localize("UTC")
                report.tz_coerced = True
                report.actions.append("Localized naive timestamps to UTC.")
        else:
            df["timestamp"] = pd.to_datetime(ts, utc=True)
            report.tz_coerced = True
            report.actions.append("Parsed and localized timestamp column to UTC.")

        return df

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        removed = before - len(df)
        report.duplicates_removed = removed
        if removed:
            report.actions.append(f"Removed {removed} duplicate timestamp rows.")
        return df

    @staticmethod
    def _sort_timestamps(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        if not df["timestamp"].is_monotonic_increasing:
            df = df.sort_values("timestamp")
            report.rows_sorted = True
            report.actions.append("Sorted rows by timestamp ascending.")
        return df

    @staticmethod
    def _drop_invalid_ohlc(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        before = len(df)
        bad = df["high"] < df["low"]
        df = df[~bad]
        removed = before - len(df)
        report.invalid_ohlc_removed = removed
        if removed:
            report.actions.append(
                f"Dropped {removed} candles where high < low (corrupt price data)."
            )
        return df

    @staticmethod
    def _drop_zero_prices(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        before = len(df)
        bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        df = df[~bad]
        removed = before - len(df)
        report.zero_price_removed = removed
        if removed:
            report.actions.append(
                f"Dropped {removed} candles with zero or negative prices."
            )
        return df
