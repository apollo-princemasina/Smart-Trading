"""
src.dataset — Dataset Builder
==============================
Assembles the final machine learning dataset by joining selected engineered
features with selected forward-looking labels.

This module is the last stage before model training.  It knows nothing about
any specific ML algorithm — it simply produces a clean, validated, versioned
DataFrame that any supervised learning algorithm can consume.

Quick start
-----------
    from src.dataset import DatasetBuilder, DatasetConfig

    config = DatasetConfig(
        symbol        = "EURUSD",
        feature_set   = "top50",
        label_groups  = ["market_bias", "trade_outcome"],
        primary_target= "direction_1b",
    )
    result = DatasetBuilder().build(config)

    df   = result.dataset        # pd.DataFrame — features + labels
    path = result.parquet_path   # data/ml/EURUSD/training_dataset_EURUSD_v1.parquet

Building from pre-loaded DataFrames (no file I/O)
--------------------------------------------------
    result = DatasetBuilder().build_from_dataframes(
        features=feature_df,
        labels=label_df,
        config=config,
    )

Output directory layout
-----------------------
data/ml/{symbol}/
    training_dataset_{symbol}_v{N}.parquet
    training_dataset_{symbol}_v{N}.csv         (if "csv" in output_formats)
    training_dataset_{symbol}_v{N}_metadata.json

reports/dataset/
    training_dataset_report.md
    training_dataset_metadata.json
"""

from .dataset_builder import (
    DatasetBuilder,
    DatasetConfig,
    DatasetResult,
)
from .dataset_loader import (
    DatasetLoader,
    LABEL_GROUP_PREFIXES,
)
from .dataset_validator import (
    DatasetValidator,
    DatasetValidatorConfig,
    DatasetValidationReport,
    ValidationIssue,
)
from .dataset_metadata import (
    DatasetMeta,
    ColumnSummary,
)
from .dataset_reports import DatasetReportGenerator

__all__ = [
    # Builder
    "DatasetBuilder",
    "DatasetConfig",
    "DatasetResult",
    # Loader
    "DatasetLoader",
    "LABEL_GROUP_PREFIXES",
    # Validator
    "DatasetValidator",
    "DatasetValidatorConfig",
    "DatasetValidationReport",
    "ValidationIssue",
    # Metadata
    "DatasetMeta",
    "ColumnSummary",
    # Reports
    "DatasetReportGenerator",
]
