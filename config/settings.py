"""Central project configuration.

All credentials are loaded from the .env file in the project root.
Never hardcode secrets here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
LOG_DIR         = BASE_DIR / "logs"
MODEL_STORE_DIR = BASE_DIR / os.getenv("MODEL_STORE", "model_store")

# Create directories on import so nothing else needs to guard them
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── MT5 credentials (loaded from .env) ───────────────────────────────────────
# Add to your .env:
#   MT5_LOGIN=12345678
#   MT5_PASSWORD=your_password
#   MT5_SERVER=ICMarkets-Demo
_raw_login   = os.getenv("MT5_LOGIN", "0")
MT5_LOGIN    = int(_raw_login) if _raw_login.isdigit() else 0
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER",   "")

# ── Supported timeframes ──────────────────────────────────────────────────────
# Matches MT5 timeframe string identifiers used throughout the project.
SUPPORTED_TIMEFRAMES: list[str] = ["W1", "D1", "H4", "H1", "M15"]

# ── Preprocessing directories ─────────────────────────────────────────────────
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORT_DIR         = BASE_DIR / "reports"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature Engineering directories ───────────────────────────────────────────
FEATURE_DIR        = DATA_DIR / "features"
FEATURE_CACHE_DIR  = DATA_DIR / "feature_cache"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature Engineering settings ──────────────────────────────────────────────
ENABLE_FEATURE_CACHE:    bool      = os.getenv("ENABLE_FEATURE_CACHE",    "true").lower() == "true"
ENABLE_PARALLEL_FEATURES: bool     = os.getenv("ENABLE_PARALLEL_FEATURES", "false").lower() == "true"
SUPPORTED_FEATURE_CATEGORIES: list[str] = [
    "market_structure",
    "liquidity",
    "sessions",
    "trend",
    "volatility",
    "momentum",
    "volume",
    "labels",
]

# ── Feature Store ─────────────────────────────────────────────────────────────
FEATURE_STORE_DIR     = DATA_DIR / os.getenv("FEATURE_STORE_DIR",   "features")
SCHEMA_DIR            = DATA_DIR / os.getenv("SCHEMA_DIR",           "schemas")
FEATURE_CATALOG_DIR   = DATA_DIR / os.getenv("FEATURE_CATALOG_DIR", "catalogs")
MANIFEST_DIR          = DATA_DIR / os.getenv("MANIFEST_DIR",         "manifests")

FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Feature Store behaviour flags
DATASET_VERSIONING:      bool = os.getenv("DATASET_VERSIONING",      "true").lower() == "true"
ENABLE_SCHEMA_HASHING:   bool = os.getenv("ENABLE_SCHEMA_HASHING",   "true").lower() == "true"
ENABLE_SCHEMA_VALIDATION: bool = os.getenv("ENABLE_SCHEMA_VALIDATION","true").lower() == "true"

# Semantic version pinned to the current schema contract
CURRENT_SCHEMA_VERSION: str = os.getenv("CURRENT_SCHEMA_VERSION", "1.0.0")
PIPELINE_VERSION:        str = os.getenv("PIPELINE_VERSION",        "1.0.0")

# ── Feature Quality Analysis ──────────────────────────────────────────────────
QUALITY_REPORT_DIR     = BASE_DIR / os.getenv("QUALITY_REPORT_DIR", "reports")
QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
CORRELATION_THRESHOLD:       float = float(os.getenv("CORRELATION_THRESHOLD",        "0.95"))
VIF_THRESHOLD:               float = float(os.getenv("VIF_THRESHOLD",                "10.0"))
PSI_THRESHOLD:               float = float(os.getenv("PSI_THRESHOLD",                "0.20"))
LEAKAGE_CORRELATION_THRESHOLD: float = float(os.getenv("LEAKAGE_CORRELATION_THRESHOLD", "0.90"))
MISSING_VALUE_THRESHOLD:     float = float(os.getenv("MISSING_VALUE_THRESHOLD",      "0.30"))
VARIANCE_THRESHOLD:          float = float(os.getenv("VARIANCE_THRESHOLD",           "1e-5"))

# Feature selection
FEATURE_SELECTION_STRATEGY:  str  = os.getenv("FEATURE_SELECTION_STRATEGY", "voting")
MIN_SELECTION_VOTES:         int  = int(os.getenv("MIN_SELECTION_VOTES",     "2"))
TOP_FEATURE_COUNTS:          list = [25, 50, 75, 100, 150]
MAX_IMPORTANCE_SAMPLES:      int  = int(os.getenv("MAX_IMPORTANCE_SAMPLES",  "50000"))

# Skip expensive steps via env (useful for CI)
SKIP_BORUTA:     bool = os.getenv("SKIP_BORUTA",     "false").lower() == "true"
SKIP_RFE:        bool = os.getenv("SKIP_RFE",        "false").lower() == "true"
SKIP_SHAP:       bool = os.getenv("SKIP_SHAP",       "false").lower() == "true"
SKIP_STABILITY:  bool = os.getenv("SKIP_STABILITY",  "false").lower() == "true"

# ── Label Generation Engine ───────────────────────────────────────────────────
LABEL_DIR             = DATA_DIR / os.getenv("LABEL_DIR", "labels")
LABEL_REPORT_DIR      = BASE_DIR / os.getenv("LABEL_REPORT_DIR", "reports") / "labels"
LABEL_DIR.mkdir(parents=True, exist_ok=True)
LABEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Trade simulation defaults
LABEL_ATR_PERIOD:      int   = int(os.getenv("LABEL_ATR_PERIOD",   "14"))
LABEL_TP_ATR_MULT:     float = float(os.getenv("LABEL_TP_ATR_MULT", "2.0"))
LABEL_SL_ATR_MULT:     float = float(os.getenv("LABEL_SL_ATR_MULT", "1.0"))
LABEL_MAX_BARS:        int   = int(os.getenv("LABEL_MAX_BARS",      "50"))

# Market bias horizons (bars)
LABEL_BIAS_HORIZONS:   list  = [1, 3, 5, 10]
LABEL_NEUTRAL_THRESHOLD: float = float(os.getenv("LABEL_NEUTRAL_THRESHOLD", "0.0003"))

# Entry timing
LABEL_TIMING_WINDOW:   int   = int(os.getenv("LABEL_TIMING_WINDOW", "10"))
LABEL_ENTER_THRESHOLD: float = float(os.getenv("LABEL_ENTER_THRESHOLD", "0.80"))

# Trade management
LABEL_TRAIL_THRESHOLD:   float = float(os.getenv("LABEL_TRAIL_THRESHOLD",   "0.50"))
LABEL_PARTIAL_THRESHOLD: float = float(os.getenv("LABEL_PARTIAL_THRESHOLD", "0.30"))

LABEL_LOG_PATH = LOG_DIR / "label_generation.log"

# ── Dataset Builder ───────────────────────────────────────────────────────────
ML_DATASET_DIR      = DATA_DIR / os.getenv("ML_DATASET_DIR",      "ml")
DATASET_REPORT_DIR  = BASE_DIR / os.getenv("DATASET_REPORT_DIR",  "reports") / "dataset"
ML_DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATASET_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Default feature / label sets
DEFAULT_FEATURE_SET:    str  = os.getenv("DEFAULT_FEATURE_SET",  "top50")
DEFAULT_LABEL_GROUPS:   list = ["market_bias", "trade_outcome"]
DEFAULT_OUTPUT_FORMATS: list = ["parquet"]
DATASET_VALIDATION_ENABLED: bool = os.getenv("DATASET_VALIDATION", "true").lower() == "true"
DATASET_MIN_ROWS:       int  = int(os.getenv("DATASET_MIN_ROWS", "100"))
DATASET_LOG_PATH        = LOG_DIR / "dataset_builder.log"

# ── Baseline Model Training ───────────────────────────────────────────────────
MODELS_DIR           = BASE_DIR / os.getenv("MODELS_DIR",    "models")
TRAINING_REPORT_DIR  = BASE_DIR / "reports" / "training"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_TARGET_COLUMN: str  = os.getenv("TRAINING_TARGET_COLUMN", "direction_1b")
TRAINING_TASK_TYPE:      str  = os.getenv("TRAINING_TASK_TYPE",     "auto")
TRAINING_RANDOM_SEED:    int  = int(os.getenv("TRAINING_RANDOM_SEED", "42"))
TRAINING_N_JOBS:         int  = int(os.getenv("TRAINING_N_JOBS",     "-1"))
TRAINING_SKIP_ON_ERROR:  bool = os.getenv("TRAINING_SKIP_ON_ERROR", "true").lower() == "true"
TRAINING_LOG_PATH       = LOG_DIR / "training.log"

# ── Walk-Forward Dataset Generator ───────────────────────────────────────────
WALK_FORWARD_DIR     = ML_DATASET_DIR / "windows"
WALK_FORWARD_REPORT_DIR = BASE_DIR / "reports" / "walk_forward"
WALK_FORWARD_DIR.mkdir(parents=True, exist_ok=True)
WALK_FORWARD_REPORT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_TYPE:        str   = os.getenv("WINDOW_TYPE",         "rolling")
TRAIN_YEARS:        int   = int(os.getenv("TRAIN_YEARS",     "5"))
VALIDATION_YEARS:   float = float(os.getenv("VALIDATION_YEARS", "1"))
TEST_YEARS:         float = float(os.getenv("TEST_YEARS",    "1"))
ROLL_FORWARD_STEP:  str   = os.getenv("ROLL_FORWARD_STEP",   "1y")
MINIMUM_TRAIN_SAMPLES: int = int(os.getenv("MINIMUM_TRAIN_SAMPLES", "100"))
MINIMUM_VAL_SAMPLES:   int = int(os.getenv("MINIMUM_VAL_SAMPLES",   "50"))
MINIMUM_TEST_SAMPLES:  int = int(os.getenv("MINIMUM_TEST_SAMPLES",  "50"))
WALK_FORWARD_GAP_BARS: int = int(os.getenv("WALK_FORWARD_GAP_BARS", "0"))
WALK_FORWARD_MAX_WINDOWS: int = int(os.getenv("WALK_FORWARD_MAX_WINDOWS", "0"))
WALK_FORWARD_LOG_PATH = LOG_DIR / "walk_forward.log"

# ── Logging ───────────────────────────────────────────────────────────────────
INGESTION_LOG_PATH        = LOG_DIR / "ingestion.log"
PREPROCESSING_LOG_PATH    = LOG_DIR / "preprocessing.log"
FEATURE_STORE_LOG_PATH    = LOG_DIR / "feature_store.log"
FEATURE_QUALITY_LOG_PATH  = LOG_DIR / "feature_quality.log"
LOG_LEVEL                 = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Hyperparameter Optimization ───────────────────────────────────────────────
OPTIMIZATION_MODELS_DIR   = MODELS_DIR
OPTIMIZATION_REPORT_DIR   = BASE_DIR / "reports" / "optimization"
OPTIMIZATION_BEST_DIR     = MODELS_DIR / "best_model"
OPTIMIZATION_STORAGE_DIR  = BASE_DIR / "optuna_storage"
OPTIMIZATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
OPTIMIZATION_BEST_DIR.mkdir(parents=True, exist_ok=True)

OPTIMIZATION_N_TRIALS:              int   = int(os.getenv("OPTIMIZATION_N_TRIALS",              "50"))
OPTIMIZATION_TIMEOUT:               float = float(os.getenv("OPTIMIZATION_TIMEOUT",             "0")) or None
OPTIMIZATION_METRIC:                str   = os.getenv("OPTIMIZATION_METRIC",                    "f1")
OPTIMIZATION_DIRECTION:             str   = os.getenv("OPTIMIZATION_DIRECTION",                 "maximize")
OPTIMIZATION_N_JOBS_TRIALS:         int   = int(os.getenv("OPTIMIZATION_N_JOBS_TRIALS",         "1"))
OPTIMIZATION_RANDOM_SEED:           int   = int(os.getenv("OPTIMIZATION_RANDOM_SEED",           "42"))
OPTIMIZATION_N_JOBS_MODEL:          int   = int(os.getenv("OPTIMIZATION_N_JOBS_MODEL",          "-1"))
OPTIMIZATION_EARLY_STOP_PATIENCE:   int   = int(os.getenv("OPTIMIZATION_EARLY_STOP_PATIENCE",   "20"))
OPTIMIZATION_EARLY_STOP_WARMUP:     int   = int(os.getenv("OPTIMIZATION_EARLY_STOP_WARMUP",     "10"))
OPTIMIZATION_EARLY_STOP_MIN_DELTA:  float = float(os.getenv("OPTIMIZATION_EARLY_STOP_MIN_DELTA","0.0001"))
OPTIMIZATION_USE_PRUNING:           bool  = os.getenv("OPTIMIZATION_USE_PRUNING",  "false").lower() == "true"
OPTIMIZATION_RESUME_IF_EXISTS:      bool  = os.getenv("OPTIMIZATION_RESUME_IF_EXISTS", "true").lower() == "true"
OPTIMIZATION_SKIP_ON_ERROR:         bool  = os.getenv("OPTIMIZATION_SKIP_ON_ERROR", "true").lower() == "true"
OPTIMIZATION_USE_SQLITE:            bool  = os.getenv("OPTIMIZATION_USE_SQLITE",    "false").lower() == "true"
OPTIMIZATION_LOG_PATH               = LOG_DIR / "optimization.log"

# ── Walk-Forward Validation Engine ───────────────────────────────────────────
VALIDATION_OUTPUT_DIR   = BASE_DIR / "validation_results"
VALIDATION_REPORT_DIR   = BASE_DIR / "reports" / "validation"
VALIDATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Acceptance thresholds
VALIDATION_MIN_ACCURACY:             float = float(os.getenv("VALIDATION_MIN_ACCURACY",             "0.50"))
VALIDATION_MIN_F1:                   float = float(os.getenv("VALIDATION_MIN_F1",                   "0.40"))
VALIDATION_MIN_DIRECTIONAL_ACCURACY: float = float(os.getenv("VALIDATION_MIN_DIRECTIONAL_ACCURACY", "0.50"))
VALIDATION_MIN_TRADING_ACCURACY:     float = float(os.getenv("VALIDATION_MIN_TRADING_ACCURACY",     "0.45"))
VALIDATION_MAX_VARIANCE:             float = float(os.getenv("VALIDATION_MAX_VARIANCE",             "0.25"))
VALIDATION_STABILITY_THRESHOLD:      float = float(os.getenv("VALIDATION_STABILITY_THRESHOLD",      "0.65"))
VALIDATION_OVERFITTING_THRESHOLD:    float = float(os.getenv("VALIDATION_OVERFITTING_THRESHOLD",    "0.15"))
VALIDATION_SKIP_ON_ERROR:            bool  = os.getenv("VALIDATION_SKIP_ON_ERROR", "true").lower() == "true"
VALIDATION_LOG_PATH                  = LOG_DIR / "validation.log"

# ── Backtesting Engine ────────────────────────────────────────────────────────
BACKTESTING_OUTPUT_DIR  = BASE_DIR / "backtesting"
BACKTESTING_REPORT_DIR  = BASE_DIR / "reports" / "backtesting"
BACKTESTING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKTESTING_REPORT_DIR.mkdir(parents=True, exist_ok=True)

BACKTESTING_INITIAL_CAPITAL:    float = float(os.getenv("BACKTESTING_INITIAL_CAPITAL",     "10000.0"))
BACKTESTING_MIN_PROBABILITY:    float = float(os.getenv("BACKTESTING_MIN_PROBABILITY",     "0.60"))
BACKTESTING_SYMBOL:             str   = os.getenv("BACKTESTING_SYMBOL",                    "EURUSD")

# Execution simulation
BACKTESTING_SPREAD_PIPS:         float = float(os.getenv("BACKTESTING_SPREAD_PIPS",         "2.0"))
BACKTESTING_COMMISSION_PER_LOT:  float = float(os.getenv("BACKTESTING_COMMISSION_PER_LOT",  "7.0"))
BACKTESTING_SLIPPAGE_PIPS:       float = float(os.getenv("BACKTESTING_SLIPPAGE_PIPS",       "0.5"))
BACKTESTING_SLIPPAGE_STD:        float = float(os.getenv("BACKTESTING_SLIPPAGE_STD",        "0.3"))
BACKTESTING_EXECUTION_DELAY_BARS: int  = int(os.getenv("BACKTESTING_EXECUTION_DELAY_BARS",  "1"))
BACKTESTING_PIP_SIZE:            float = float(os.getenv("BACKTESTING_PIP_SIZE",            "0.0001"))
BACKTESTING_PIP_VALUE:           float = float(os.getenv("BACKTESTING_PIP_VALUE",           "10.0"))

# SL/TP
BACKTESTING_SL_TP_MODE:          str   = os.getenv("BACKTESTING_SL_TP_MODE",                "atr")
BACKTESTING_SL_PIPS:             float = float(os.getenv("BACKTESTING_SL_PIPS",             "20.0"))
BACKTESTING_TP_PIPS:             float = float(os.getenv("BACKTESTING_TP_PIPS",             "40.0"))
BACKTESTING_SL_ATR_MULT:         float = float(os.getenv("BACKTESTING_SL_ATR_MULT",         "1.5"))
BACKTESTING_TP_ATR_MULT:         float = float(os.getenv("BACKTESTING_TP_ATR_MULT",         "3.0"))
BACKTESTING_ENABLE_TRAILING:     bool  = os.getenv("BACKTESTING_ENABLE_TRAILING", "false").lower() == "true"
BACKTESTING_TRAILING_PIPS:       float = float(os.getenv("BACKTESTING_TRAILING_PIPS",       "20.0"))
BACKTESTING_ENABLE_BREAK_EVEN:   bool  = os.getenv("BACKTESTING_ENABLE_BREAK_EVEN", "true").lower() == "true"
BACKTESTING_BE_TRIGGER_RR:       float = float(os.getenv("BACKTESTING_BE_TRIGGER_RR",       "1.0"))
BACKTESTING_ENABLE_TIME_STOP:    bool  = os.getenv("BACKTESTING_ENABLE_TIME_STOP", "false").lower() == "true"
BACKTESTING_MAX_HOLDING_BARS:    int   = int(os.getenv("BACKTESTING_MAX_HOLDING_BARS",      "48"))

# Position sizing
BACKTESTING_POSITION_MODE:       str   = os.getenv("BACKTESTING_POSITION_MODE",             "fixed_risk_pct")
BACKTESTING_FIXED_LOT_SIZE:      float = float(os.getenv("BACKTESTING_FIXED_LOT_SIZE",      "0.10"))
BACKTESTING_RISK_PCT:            float = float(os.getenv("BACKTESTING_RISK_PCT",            "0.01"))
BACKTESTING_MIN_LOT:             float = float(os.getenv("BACKTESTING_MIN_LOT",             "0.01"))
BACKTESTING_MAX_LOT:             float = float(os.getenv("BACKTESTING_MAX_LOT",             "10.0"))

# Risk management
BACKTESTING_MAX_OPEN_POSITIONS:  int   = int(os.getenv("BACKTESTING_MAX_OPEN_POSITIONS",    "3"))
BACKTESTING_MAX_DAILY_LOSS_PCT:  float = float(os.getenv("BACKTESTING_MAX_DAILY_LOSS_PCT",  "0.02"))
BACKTESTING_MAX_WEEKLY_LOSS_PCT: float = float(os.getenv("BACKTESTING_MAX_WEEKLY_LOSS_PCT", "0.05"))

BACKTESTING_SKIP_ON_ERROR:       bool  = os.getenv("BACKTESTING_SKIP_ON_ERROR", "true").lower() == "true"
BACKTESTING_RANDOM_SEED:         int   = int(os.getenv("BACKTESTING_RANDOM_SEED",           "42"))
BACKTESTING_LOG_PATH             = LOG_DIR / "backtesting.log"

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'database.db'}",
)
