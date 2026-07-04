"""
Walk-Forward Dataset Generator
==============================
Creates rolling chronological train/validation/test splits for
realistic time-series machine learning evaluation.

Guaranteed properties
---------------------
* Strict chronological ordering — data is NEVER shuffled.
* No row appears in more than one split.
* Zero look-ahead bias — train rows always precede val/test rows.
* Gap bars between splits eliminate label-overlap leakage.

Quick start
-----------
    from src.walk_forward import WalkForwardGenerator, WalkForwardConfig

    gen = WalkForwardGenerator()
    result = gen.run(
        dataset="data/ml/EURUSD/training_dataset_EURUSD_v1.parquet",
        symbol="EURUSD",
        config=WalkForwardConfig(
            window_type  = "rolling",
            train_period = "5y",
            val_period   = "1y",
            test_period  = "1y",
            step_period  = "1y",
        ),
    )
    print(result)
"""
from .dataset_splitter import DatasetSplitter, SplitResult
from .reports import generate_walk_forward_report
from .split_validator import (
    FAIL,
    PASS,
    WARNING,
    SplitIssue,
    SplitValidationReport,
    SplitValidator,
    SplitValidatorConfig,
)
from .walk_forward_generator import (
    WalkForwardConfig,
    WalkForwardGenerator,
    WalkForwardResult,
)
from .window_generator import (
    WindowConfig,
    WindowGenerator,
    WindowSpec,
    _first_bar_at_or_after,
    _last_bar_before,
    parse_period,
)
from .window_metadata import SplitStats, WindowMeta

__all__ = [
    # Generator
    "WalkForwardGenerator",
    "WalkForwardConfig",
    "WalkForwardResult",
    # Window generation
    "WindowGenerator",
    "WindowConfig",
    "WindowSpec",
    "parse_period",
    "_last_bar_before",
    "_first_bar_at_or_after",
    # Splitting
    "DatasetSplitter",
    "SplitResult",
    # Validation
    "SplitValidator",
    "SplitValidatorConfig",
    "SplitValidationReport",
    "SplitIssue",
    "PASS",
    "WARNING",
    "FAIL",
    # Metadata
    "WindowMeta",
    "SplitStats",
    # Reports
    "generate_walk_forward_report",
]
