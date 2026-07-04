"""Run the feature engineering pipeline for EURUSD.

Loads the merged M15 OHLCV from data/processed/EURUSD/merged/,
executes all registered feature generators,
saves the feature dataset to data/features/EURUSD/feature_dataset.parquet,
and generates reports/feature_pipeline_report.md.

Usage
-----
    python scripts/run_feature_pipeline.py
    python scripts/run_feature_pipeline.py --symbol EURUSD
    python scripts/run_feature_pipeline.py --symbol EURUSD --no-cache
    python scripts/run_feature_pipeline.py --disable volatility_placeholder

No network connections are made — works entirely from local Parquet files.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 rejects many Unicode characters).
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

# Ensure project root is importable.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from config.settings import (
    PROCESSED_DATA_DIR,
    FEATURE_DIR,
    FEATURE_CACHE_DIR,
    REPORT_DIR,
    ENABLE_FEATURE_CACHE,
    ENABLE_PARALLEL_FEATURES,
    LOG_DIR,
    LOG_LEVEL,
    PREPROCESSING_LOG_PATH,
)

# ── Import the features package — this triggers auto-registration ─────────────
import src.features as features_pkg
from src.features import FeaturePipeline, FeatureRegistry

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
_feature_log_path = LOG_DIR / "feature_engineering.log"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_feature_log_path, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ICT + ML feature engineering pipeline — "
            "validates, executes, and saves all feature generators."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Instrument symbol to process.",
    )
    parser.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="Disable feature caching (ignore existing cache files).",
    )
    parser.add_argument(
        "--disable",
        nargs="*",
        metavar="FEATURE_NAME",
        default=[],
        help=(
            "Feature names to disable for this run. "
            "Example: --disable volatility_placeholder momentum_placeholder"
        ),
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print all registered features and exit.",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args   = _parse_args()
    symbol = args.symbol.upper()

    # Disable requested features
    for name in (args.disable or []):
        try:
            FeatureRegistry.disable(name)
            logger.info("Disabled feature: %s", name)
        except KeyError:
            logger.warning("Cannot disable '%s' — not registered.", name)

    # List-and-exit mode
    if args.list_features:
        print(FeatureRegistry.summary())
        return

    use_cache = ENABLE_FEATURE_CACHE and not args.no_cache

    logger.info("Symbol            : %s", symbol)
    logger.info("Processed dir     : %s", PROCESSED_DATA_DIR)
    logger.info("Feature dir       : %s", FEATURE_DIR)
    logger.info("Cache dir         : %s", FEATURE_CACHE_DIR)
    logger.info("Report dir        : %s", REPORT_DIR)
    logger.info("Cache enabled     : %s", use_cache)
    logger.info("Parallel features : %s", ENABLE_PARALLEL_FEATURES)
    logger.info("Log file          : %s", _feature_log_path)
    logger.info("")
    logger.info(FeatureRegistry.summary())

    pipeline = FeaturePipeline(
        processed_dir   = PROCESSED_DATA_DIR,
        feature_dir     = FEATURE_DIR,
        report_dir      = REPORT_DIR,
        cache_dir       = FEATURE_CACHE_DIR,
        enable_cache    = use_cache,
        enable_parallel = ENABLE_PARALLEL_FEATURES,
    )

    dataset_path = pipeline.run(symbol)

    logger.info("")
    logger.info("Done.")
    logger.info("  Feature dataset : %s", dataset_path)
    logger.info("  Pipeline report : %s", REPORT_DIR / "feature_pipeline_report.md")
    logger.info("  Log file        : %s", _feature_log_path)


if __name__ == "__main__":
    main()
