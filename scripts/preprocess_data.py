"""Run the full data preprocessing pipeline for EURUSD.

Reads raw Parquet files from data/raw/EURUSD/{TF}/,
applies validation + cleaning + market-calendar checks,
runs cross-timeframe consistency validation,
merges all timeframes onto the M15 base (no lookahead),
saves processed files to data/processed/EURUSD/{TF}/,
and writes a quality report to reports/data_quality_report.md.

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --symbol EURUSD
    python scripts/preprocess_data.py --symbol EURUSD --timeframes D1 H4 H1 M15

No external data connections are made — works entirely from local files.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows (console defaults to cp1252)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    PREPROCESSING_LOG_PATH,
    SUPPORTED_TIMEFRAMES,
    LOG_LEVEL,
    LOG_DIR,
)
from src.preprocessing import PreprocessingPipeline

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PREPROCESSING_LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICT + ML preprocessing pipeline — validates, cleans, and merges OHLCV data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Instrument symbol to process.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=SUPPORTED_TIMEFRAMES,
        choices=SUPPORTED_TIMEFRAMES,
        metavar="TF",
        help=f"Timeframes to process. Choices: {SUPPORTED_TIMEFRAMES}",
    )
    parser.add_argument(
        "--base-tf",
        default="M15",
        choices=SUPPORTED_TIMEFRAMES,
        help="Base (lowest) timeframe used for the multi-TF merge.",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args   = _parse_args()
    symbol = args.symbol.upper()
    tfs    = [tf.upper() for tf in args.timeframes]
    base   = args.base_tf.upper()

    # Warn if base TF not in the list
    if base not in tfs:
        logger.warning(
            "Base timeframe %s is not in the timeframe list %s. "
            "Multi-TF merge will be skipped.",
            base, tfs,
        )

    logger.info("Symbol     : %s", symbol)
    logger.info("Timeframes : %s", tfs)
    logger.info("Base TF    : %s", base)
    logger.info("Raw dir    : %s", RAW_DATA_DIR)
    logger.info("Proc dir   : %s", PROCESSED_DATA_DIR)
    logger.info("Report dir : %s", REPORT_DIR)

    pipeline = PreprocessingPipeline(
        raw_data_dir  = RAW_DATA_DIR,
        processed_dir = PROCESSED_DATA_DIR,
        report_dir    = REPORT_DIR,
        timeframes    = tfs,
        base_tf       = base,
    )

    report_path = pipeline.run(symbol)

    logger.info("")
    logger.info("Done. Quality report: %s", report_path)


if __name__ == "__main__":
    main()
