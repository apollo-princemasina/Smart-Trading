"""Incremental update for MT5 Parquet datasets.

Reads the most recent Parquet file for a symbol/timeframe, determines
the last recorded candle timestamp, downloads only the missing candles
from MT5, appends them, deduplicates, and saves the updated file.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .mt5_downloader import MT5Downloader, MT5Config
from .validate_data import validate_dataframe
from config.settings import (
    RAW_DATA_DIR,
    SUPPORTED_TIMEFRAMES,
    INGESTION_LOG_PATH,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.FileHandler(INGESTION_LOG_PATH, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_h)


class MT5DataUpdater:
    """
    Append new candles to an existing MT5 Parquet dataset.

    Designed for daily/live incremental updates after the initial historical
    backfill has been completed via download_historical_data.py.

    Usage
    -----
    >>> updater = MT5DataUpdater()
    >>> path    = updater.update("EURUSD", "H1")
    """

    def __init__(self, downloader: Optional[MT5Downloader] = None) -> None:
        if downloader is not None:
            self._dl = downloader
        else:
            cfg      = MT5Config(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
            self._dl = MT5Downloader(cfg)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        symbol:    str,
        timeframe: str,
        end_date:  Optional[datetime] = None,
    ) -> Path:
        """
        Append missing candles to the latest Parquet file for symbol/timeframe.

        Args:
            symbol:    Instrument symbol (e.g. "EURUSD").
            timeframe: One of W1 | D1 | H4 | H1 | M15.
            end_date:  Inclusive end date for the update window.
                       Defaults to today (UTC midnight).

        Returns:
            Path to the updated Parquet file.

        Raises:
            FileNotFoundError: No existing Parquet file found — run the
                               historical download script first.
            ValueError:        Unsupported timeframe.
            RuntimeError:      Validation failure after merge.
        """
        symbol    = symbol.upper()
        timeframe = timeframe.upper()

        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Valid options: {SUPPORTED_TIMEFRAMES}"
            )

        existing_path = self._find_latest_file(symbol, timeframe)
        logger.info("Updating %s %s using %s", symbol, timeframe, existing_path.name)

        existing = self._load_parquet(existing_path)
        last_ts  = existing["timestamp"].max()

        if pd.isna(last_ts):
            raise ValueError(
                f"Existing file {existing_path} contains no valid timestamps."
            )

        step       = MT5Downloader.granularity_timedelta(timeframe)  # type: ignore[attr-defined]
        next_start = last_ts + step
        download_end = (
            pd.Timestamp(end_date, tz="UTC")
            if end_date is not None
            else pd.Timestamp.utcnow().normalize().tz_localize("UTC")
        )

        if next_start >= download_end:
            logger.info("%s %s is already up to date.", symbol, timeframe)
            return existing_path

        logger.info(
            "Downloading new candles | symbol=%s timeframe=%s from=%s to=%s",
            symbol, timeframe,
            next_start.strftime("%Y-%m-%d %H:%M"),
            download_end.strftime("%Y-%m-%d %H:%M"),
        )

        self._dl.initialize()
        try:
            self._dl.login()
            new_df = self._dl.download(
                symbol     = symbol,
                timeframe  = timeframe,
                start_date = next_start.to_pydatetime(),
                end_date   = download_end.to_pydatetime(),
            )
        finally:
            self._dl.shutdown()

        if new_df.empty:
            logger.info("No new candles available for %s %s.", symbol, timeframe)
            return existing_path

        combined = (
            pd.concat([existing, new_df], ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        report = validate_dataframe(combined, timeframe)
        if not report.passed:
            logger.error("Post-update validation failed: %s", report.messages)
            raise RuntimeError(
                f"Merged dataset failed validation — not saved. "
                f"Issues: {report.messages}"
            )

        output_path = self._get_output_path(symbol, timeframe, combined)
        combined.to_parquet(output_path, index=False, engine="pyarrow")

        if output_path != existing_path:
            existing_path.unlink(missing_ok=True)

        logger.info(
            "Update complete | symbol=%s timeframe=%s rows=%d path=%s",
            symbol, timeframe, len(combined), output_path.name,
        )
        return output_path

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _find_latest_file(symbol: str, timeframe: str) -> Path:
        directory = RAW_DATA_DIR / symbol / timeframe
        if not directory.exists():
            raise FileNotFoundError(
                f"No data directory found at {directory}. "
                "Run download_historical_data.py first."
            )
        candidates = sorted(directory.glob(f"{symbol}_{timeframe}_*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No Parquet files found in {directory}."
            )
        return max(candidates, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _load_parquet(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return (
            df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    @staticmethod
    def _get_output_path(symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
        start_year = int(df["timestamp"].dt.year.min())
        end_year   = int(df["timestamp"].dt.year.max())
        directory  = RAW_DATA_DIR / symbol / timeframe
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{symbol}_{timeframe}_{start_year}_{end_year}.parquet"
