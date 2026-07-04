"""Resumable full ingestion runner for MT5 historical data (year-by-year).

Saves a manifest to data/ingestion_manifest.json so interrupted runs can
resume without re-downloading completed years.

Usage:
    python scripts/run_full_ingest.py
    python scripts/run_full_ingest.py --symbol EURUSD --timeframe M15
    python scripts/run_full_ingest.py --start-year 2020 --end-year 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from config.settings import (
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    RAW_DATA_DIR,
    SUPPORTED_TIMEFRAMES,
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

# ── Manifest path ─────────────────────────────────────────────────────────────
MANIFEST = ROOT / "data" / "ingestion_manifest.json"


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt manifest — starting fresh.")
    return {}


def _save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Year-by-year resumable MT5 ingestion with manifest tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol",     default="EURUSD", help="Instrument symbol.")
    parser.add_argument(
        "--timeframe",
        default="M15",
        choices=SUPPORTED_TIMEFRAMES,
        help="Timeframe to download.",
    )
    parser.add_argument("--start-year", type=int, default=2017, help="First year to collect.")
    parser.add_argument("--end-year",   type=int, default=2025, help="Last year to collect (inclusive).")
    return parser.parse_args()


# ── Core runner ───────────────────────────────────────────────────────────────

def run(
    symbol:     str,
    timeframe:  str,
    start_year: int,
    end_year:   int,
) -> None:
    """
    Download one year at a time, tracking progress in the manifest file.

    Completed years are skipped on re-runs, so the script is safe to
    interrupt and resume at any point.
    """
    cfg        = MT5Config(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    downloader = MT5Downloader(cfg)
    manifest   = _load_manifest()

    downloader.initialize()
    try:
        downloader.login()

        for year in range(start_year, end_year + 1):
            year_key = f"{symbol}_{timeframe}_{year}"

            # Skip years already marked complete in the manifest
            if manifest.get(year_key, {}).get("status") == "completed":
                logger.info("Skipping %s %s %d (already complete)", symbol, timeframe, year)
                continue

            logger.info("")
            logger.info("-- %s %s year=%d --", symbol, timeframe, year)

            manifest.setdefault(year_key, {})
            manifest[year_key].update({
                "status":     "in_progress",
                "started_at": datetime.utcnow().isoformat(),
            })
            _save_manifest(manifest)

            start_dt = datetime(year, 1, 1)
            end_dt   = datetime(year, 12, 31)

            try:
                df = downloader.download(
                    symbol     = symbol,
                    timeframe  = timeframe,
                    start_date = start_dt,
                    end_date   = end_dt,
                )

                if df.empty:
                    logger.warning("No data returned for %s %s %d", symbol, timeframe, year)
                    manifest[year_key].update({
                        "status":      "no_data",
                        "rows":        0,
                        "finished_at": datetime.utcnow().isoformat(),
                    })
                    _save_manifest(manifest)
                    continue

                report = validate_dataframe(df, timeframe)
                for msg in report.messages:
                    logger.info("  [validate] %s", msg)

                path = downloader.save(
                    df         = df,
                    symbol     = symbol,
                    timeframe  = timeframe,
                    start_year = year,
                    end_year   = year,
                )
                size_mb = path.stat().st_size / 1_048_576
                logger.info(
                    "  Saved → %s  (%.1f MB, %d rows)", path.name, size_mb, len(df)
                )

                manifest[year_key].update({
                    "status":      "completed",
                    "rows":        int(len(df)),
                    "path":        str(path),
                    "finished_at": datetime.utcnow().isoformat(),
                })
                _save_manifest(manifest)

            except Exception as exc:
                logger.exception("Failed for %s %s %d: %s", symbol, timeframe, year, exc)
                manifest[year_key].update({
                    "status":      "failed",
                    "error":       str(exc),
                    "finished_at": datetime.utcnow().isoformat(),
                })
                _save_manifest(manifest)

            # Brief pause between years to avoid hammering the MT5 server
            time.sleep(0.5)

    finally:
        downloader.shutdown()

    logger.info("")
    logger.info("Ingestion run complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    run(
        symbol     = args.symbol.upper(),
        timeframe  = args.timeframe.upper(),
        start_year = args.start_year,
        end_year   = args.end_year,
    )
