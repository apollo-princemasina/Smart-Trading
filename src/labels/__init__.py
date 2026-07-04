"""
src.labels — Label Generation Engine
======================================
Generates production-quality supervised-learning labels for five models:

  Model 1 — Market Bias          (direction, return, confidence, probability)
  Model 2 — Setup Quality        (grade 0-3, quality score)
  Model 3 — Entry Timing         (enter-now / wait / ignore)
  Model 4 — Trade Outcome        (TP/SL/timeout, MFE, MAE, RR)
  Model 5 — Trade Management     (hold/trail/scale-out/early-exit strategy)

All labels are STRICTLY forward-looking — computed from future price data and
stored in a separate directory.  They must NEVER be used as input features.

Quick start
-----------
    from src.labels import LabelPipeline

    pipeline = LabelPipeline()
    result   = pipeline.run(ohlcv_df, symbol="EURUSD")
    labels   = result.labels            # pd.DataFrame, same index as ohlcv_df
    path     = result.parquet_path      # data/labels/EURUSD/labels_EURUSD_v1.parquet
"""

from .market_bias import (
    MarketBiasLabeler,
    MarketBiasConfig,
    MarketBiasLabels,
    BEARISH,
    NEUTRAL,
    BULLISH,
)
from .trade_outcome import (
    TradeOutcomeLabeler,
    TradeOutcomeConfig,
    TradeOutcomeLabels,
    compute_atr,
    simulate_trade,
    TIMEOUT,
    TP_FIRST,
    SL_FIRST,
)
from .setup_quality import (
    SetupQualityLabeler,
    SetupQualityConfig,
    SetupQualityLabels,
    NO_TRADE,
    LOW,
    MEDIUM,
    HIGH,
)
from .entry_timing import (
    EntryTimingLabeler,
    EntryTimingConfig,
    EntryTimingLabels,
    IGNORE,
    WAIT,
    ENTER_NOW,
)
from .trade_management import (
    TradeManagementLabeler,
    TradeManagementConfig,
    TradeManagementLabels,
    SIMPLE,
    TRAIL,
    SCALE_OUT,
    EARLY_EXIT,
)
from .label_validator import (
    LabelValidator,
    LabelValidatorConfig,
    ValidationReport,
    ValidationIssue,
)
from .label_metadata import LabelMeta, ColumnMeta
from .label_reports  import LabelReportGenerator
from .label_pipeline import (
    LabelPipeline,
    LabelPipelineConfig,
    LabelPipelineResult,
)

__all__ = [
    # Pipeline
    "LabelPipeline",
    "LabelPipelineConfig",
    "LabelPipelineResult",
    # Labelers
    "MarketBiasLabeler",      "MarketBiasConfig",      "MarketBiasLabels",
    "TradeOutcomeLabeler",    "TradeOutcomeConfig",    "TradeOutcomeLabels",
    "SetupQualityLabeler",    "SetupQualityConfig",    "SetupQualityLabels",
    "EntryTimingLabeler",     "EntryTimingConfig",     "EntryTimingLabels",
    "TradeManagementLabeler", "TradeManagementConfig", "TradeManagementLabels",
    # Validation & metadata
    "LabelValidator",         "LabelValidatorConfig",
    "ValidationReport",       "ValidationIssue",
    "LabelMeta",              "ColumnMeta",
    "LabelReportGenerator",
    # Utilities
    "compute_atr",
    "simulate_trade",
    # Constants
    "BEARISH", "NEUTRAL", "BULLISH",
    "TIMEOUT", "TP_FIRST", "SL_FIRST",
    "NO_TRADE", "LOW", "MEDIUM", "HIGH",
    "IGNORE", "WAIT", "ENTER_NOW",
    "SIMPLE", "TRAIL", "SCALE_OUT", "EARLY_EXIT",
]
