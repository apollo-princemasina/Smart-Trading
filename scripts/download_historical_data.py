"""Download historical EURUSD OHLCV data from MetaTrader 5 (2017–2025).

Automatically downloads all five required timeframes:
    W1 | D1 | H4 | H1 | M15

and saves each as a separate Parquet file under data/raw/EURUSD/.

Usage:
    python scripts/download_historical_data.py

Prerequisites:
    - MetaTrader 5 terminal installed and open on this machine.
    - Credentials in .env:
          MT5_LOGIN=12345678
          MT5_PASSWORD=your_password
          MT5_SERVER=ICMarkets-Demo   (or your broker's server name)

Output files (one per timeframe):
    data/raw/EURUSD/W1/EURUSD_W1_2017_2025.parquet
    data/raw/EURUSD/D1/EURUSD_D1_2017_2025.parquet
    data/raw/EURUSD/H4/EURUSD_H4_2017_2025.parquet
    data/raw/EURUSD/H1/EURUSD_H1_2017_2025.parquet
    data/raw/EURUSD/M15/EURUSD_M15_2017_2025.parquet
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Force UTF-8 output on Windows (console defaults to cp1252 which rejects
# many Unicode characters that appear in log messages and library output).
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

# ── Ensure project root is importable ────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from config.settings import (
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    RAW_DATA_DIR,
    LOG_DIR,
)
from src.ingestion.mt5_downloader import MT5Config, MT5Downloader
from src.ingestion.validate_data import validate_dataframe

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOL     = "EURUSD"
START_DATE = datetime(2017, 1, 1)
END_DATE   = datetime(2025, 12, 31)
START_YEAR = START_DATE.year
END_YEAR   = END_DATE.year

# All five required timeframes — add more symbols/timeframes here without
# touching any other part of the code.
TIMEFRAMES: list[str] = ["W1", "D1", "H4", "H1", "M15"]


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _validate_credentials()

    cfg        = MT5Config(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    downloader = MT5Downloader(cfg)

    logger.info("=" * 65)
    logger.info("MT5 historical data download")
    logger.info("  Symbol     : %s", SYMBOL)
    logger.info("  Range      : %s -> %s", START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"))
    logger.info("  Timeframes : %s", TIMEFRAMES)
    logger.info("  Output dir : %s", RAW_DATA_DIR)
    logger.info("=" * 65)

    downloader.initialize()
    try:
        downloader.login()
        _run_downloads(downloader)
    finally:
        downloader.shutdown()


# Timeframes that must be chunked year-by-year because a single
# copy_rates_range() call over 9 years exceeds broker server limits.
CHUNKED_TIMEFRAMES: set[str] = {"M15", "H1"}


def _run_downloads(downloader: MT5Downloader) -> None:
    """Iterate over every timeframe, download, validate, and save."""
    failed: list[str] = []

    for timeframe in TIMEFRAMES:
        logger.info("")
        logger.info("-- %s %s --", SYMBOL, timeframe)

        try:
            if timeframe in CHUNKED_TIMEFRAMES:
                df = _download_chunked_by_year(downloader, timeframe)
            else:
                df = downloader.download(
                    symbol     = SYMBOL,
                    timeframe  = timeframe,
                    start_date = START_DATE,
                    end_date   = END_DATE,
                )
        except Exception:
            logger.exception("Download failed for %s %s", SYMBOL, timeframe)
            failed.append(timeframe)
            continue

        if df.empty:
            logger.warning("No data returned for %s %s -- skipping.", SYMBOL, timeframe)
            failed.append(timeframe)
            continue

        # ── Validate ──────────────────────────────────────────────────────────
        report = validate_dataframe(df, timeframe)
        for msg in report.messages:
            level = logging.INFO if report.passed else logging.WARNING
            logger.log(level, "  [validate] %s", msg)

        if not report.passed:
            logger.warning(
                "Validation warnings for %s %s -- file will still be saved.",
                SYMBOL, timeframe,
            )

        # ── Save ──────────────────────────────────────────────────────────────
        try:
            path = downloader.save(
                df         = df,
                symbol     = SYMBOL,
                timeframe  = timeframe,
                start_year = START_YEAR,
                end_year   = END_YEAR,
            )
            size_mb = path.stat().st_size / 1_048_576
            logger.info("  Saved -> %s  (%.1f MB, %d rows)", path.name, size_mb, len(df))
        except Exception:
            logger.exception("Save failed for %s %s", SYMBOL, timeframe)
            failed.append(timeframe)

    logger.info("")
    logger.info("=" * 65)
    if failed:
        logger.error("Failed timeframes: %s -- re-run to retry.", failed)
        sys.exit(1)
    else:
        logger.info("All timeframes downloaded and saved successfully.")
    logger.info("=" * 65)


def _download_chunked_by_year(downloader: MT5Downloader, timeframe: str) -> pd.DataFrame:
    """
    Download a high-frequency timeframe year-by-year and concatenate.

    MT5 brokers cap the number of bars a single copy_rates_range() call can
    return. For M15 and H1 over a 9-year range the bar count easily exceeds
    that cap, resulting in error -2 (Invalid params). Splitting by year keeps
    each request well within broker limits (~25 K bars/year for M15).
    """
    frames: list[pd.DataFrame] = []

    for year in range(START_YEAR, END_YEAR + 1):
        year_start = datetime(year, 1, 1)
        year_end   = datetime(year, 12, 31)
        logger.info("  Downloading %s %s for %d ...", SYMBOL, timeframe, year)
        try:
            chunk = downloader.download(
                symbol     = SYMBOL,
                timeframe  = timeframe,
                start_date = year_start,
                end_date   = year_end,
            )
            # MT5 returns a single filler bar when no real history exists for
            # the requested period (common on demo accounts beyond ~2-3 years).
            # Discard any chunk with fewer than 5 rows as it is not real data.
            if len(chunk) < 5:
                logger.info("    %d rows for %d -- likely beyond broker history depth, skipping.", len(chunk), year)
            else:
                logger.info("    %d rows retrieved for %d", len(chunk), year)
                frames.append(chunk)
        except Exception as exc:
            logger.error("  Year %d failed for %s %s: %s", year, SYMBOL, timeframe, exc)

    if not frames:
        return pd.DataFrame()

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    logger.info(
        "  Combined %s %s: %d total rows (%d-%d)",
        SYMBOL, timeframe, len(combined), START_YEAR, END_YEAR,
    )
    return combined


def _validate_credentials() -> None:
    """Exit early with a clear message if MT5 credentials are missing."""
    missing = [
        name
        for name, val in [
            ("MT5_LOGIN",    str(MT5_LOGIN)),
            ("MT5_PASSWORD", MT5_PASSWORD),
            ("MT5_SERVER",   MT5_SERVER),
        ]
        if not val or val == "0"
    ]
    if missing:
        logger.error(
            "Missing MT5 credentials in .env: %s\n"
            "Add them like:\n"
            "    MT5_LOGIN=12345678\n"
            "    MT5_PASSWORD=your_password\n"
            "    MT5_SERVER=ICMarkets-Demo",
            missing,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
