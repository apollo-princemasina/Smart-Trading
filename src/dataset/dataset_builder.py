"""
Dataset Builder
===============
Assembles the final machine learning dataset by joining engineered features
with forward-looking labels.

The builder is the last step before model training and knows nothing about
any specific ML algorithm.  Its sole job: produce a clean, validated,
versioned dataset that any supervised learning algorithm can consume.

No-look-ahead guarantee
-----------------------
Features are indexed at time T and represent information available AT T.
Labels are indexed at time T and represent outcomes AFTER T (computed
during label generation).  The inner join on timestamp ensures every row
has a known past (features) and a known future outcome (label).
The builder never shifts, rolls, or interpolates across time boundaries.

Usage
-----
    from src.dataset import DatasetBuilder, DatasetConfig

    config  = DatasetConfig(symbol="EURUSD", feature_set="top50",
                            label_groups=["market_bias", "trade_outcome"])
    builder = DatasetBuilder()
    result  = builder.build(config)

    # result.dataset        — pd.DataFrame (features + labels, NaN tail dropped)
    # result.parquet_path   — data/ml/EURUSD/training_dataset_EURUSD_v1.parquet
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .dataset_loader    import DatasetLoader, LABEL_GROUP_PREFIXES
from .dataset_validator import DatasetValidator, DatasetValidatorConfig, DatasetValidationReport
from .dataset_metadata  import DatasetMeta
from .dataset_reports   import DatasetReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for one dataset build run.

    Args:
        symbol:            Instrument symbol (e.g. "EURUSD").
        feature_set:       Feature subset: top25/top50/top75/top100/top150/all/custom.
        custom_features:   Column list (used when feature_set == 'custom').
        label_groups:      Label model groups to include (None = all).
        custom_labels:     Explicit label column list override.
        primary_target:    Column used for NaN-row dropping (e.g. "direction_1b").
        drop_na_labels:    Drop rows where primary_target is NaN.
        output_formats:    File formats to write: parquet and/or csv.
        output_dir:        Base output directory (default: data/ml).
        validate:          Run DatasetValidator on assembled dataset.
        min_rows:          Minimum required rows after assembly.
        feature_version:   Feature Store version (None = latest).
        label_version:     Label Store version (None = latest).
        dataset_name:      Name used in output filenames and metadata.
        prediction_timeframe: Timeframe of the prediction (e.g. "H1").
        higher_timeframes: Additional timeframes included in features.
        pipeline_version:  Semantic version tag for this build.
    """
    symbol:               str
    feature_set:          str            = "top50"
    custom_features:      Optional[list[str]] = None
    label_groups:         Optional[list[str]] = None   # None = all
    custom_labels:        Optional[list[str]] = None
    primary_target:       Optional[str]  = None
    drop_na_labels:       bool           = True
    output_formats:       list[str]      = field(default_factory=lambda: ["parquet"])
    output_dir:           Optional[Path] = None
    validate:             bool           = True
    min_rows:             int            = 100
    feature_version:      Optional[int]  = None
    label_version:        Optional[int]  = None
    dataset_name:         str            = "training_dataset"
    prediction_timeframe: str            = ""
    higher_timeframes:    list[str]      = field(default_factory=list)
    pipeline_version:     str            = "1.0.0"


@dataclass
class DatasetResult:
    """Result of a dataset build run."""
    symbol:           str
    dataset:          pd.DataFrame
    feature_columns:  list[str]
    label_columns:    list[str]
    n_rows:           int
    n_features:       int
    n_labels:         int
    parquet_path:     Optional[Path]
    csv_path:         Optional[Path]
    metadata:         DatasetMeta
    validation:       Optional[DatasetValidationReport]
    report_paths:     dict[str, Path] = field(default_factory=dict)
    build_time_s:     float = 0.0


class DatasetBuilder:
    """Assemble, validate, and persist the training dataset."""

    def __init__(
        self,
        loader:      Optional[DatasetLoader]          = None,
        validator:   Optional[DatasetValidator]       = None,
        reporter:    Optional[DatasetReportGenerator] = None,
        output_dir:  Optional[Path]                   = None,
        report_dir:  Optional[Path]                   = None,
    ) -> None:
        from config.settings import ML_DATASET_DIR, DATASET_REPORT_DIR

        self.output_dir = Path(output_dir or ML_DATASET_DIR)
        self.report_dir = Path(report_dir or DATASET_REPORT_DIR)
        self.loader     = loader    or DatasetLoader()
        self.validator  = validator or DatasetValidator()
        self.reporter   = reporter  or DatasetReportGenerator(self.report_dir)

    # ── Public API ────────────────────────────────────────────────────────

    def build(
        self,
        config:            DatasetConfig,
        feature_parquet:   Optional[Path] = None,
        label_parquet:     Optional[Path] = None,
    ) -> DatasetResult:
        """Load features + labels from stores, assemble, validate, and save.

        Args:
            config:          Build configuration.
            feature_parquet: Override: load features from this parquet (testing).
            label_parquet:   Override: load labels from this parquet (testing).

        Returns:
            DatasetResult containing the assembled dataset and all paths.
        """
        t0 = time.perf_counter()
        logger.info(
            "DatasetBuilder.build: symbol=%s  feature_set=%s  label_groups=%s",
            config.symbol, config.feature_set, config.label_groups,
        )

        features = self.loader.load_features(
            symbol=config.symbol,
            feature_set=config.feature_set,
            custom_features=config.custom_features,
            version=config.feature_version,
            parquet_path=feature_parquet,
        )
        labels = self.loader.load_labels(
            symbol=config.symbol,
            label_groups=config.label_groups,
            custom_labels=config.custom_labels,
            version=config.label_version,
            parquet_path=label_parquet,
        )

        return self._assemble_and_save(features, labels, config, t0)

    def build_from_dataframes(
        self,
        features: pd.DataFrame,
        labels:   pd.DataFrame,
        config:   DatasetConfig,
    ) -> DatasetResult:
        """Assemble from pre-loaded DataFrames (for testing / programmatic use).

        The caller is responsible for ensuring ``features`` and ``labels`` have
        matching DatetimeIndex values.  Input DataFrames are never mutated.
        """
        t0 = time.perf_counter()
        # Apply feature/label filtering
        features = self.loader._filter_feature_set(
            features.copy(), config.feature_set, config.custom_features
        )
        labels = self.loader._filter_label_groups(
            labels.copy(), config.label_groups, config.custom_labels
        )
        return self._assemble_and_save(features, labels, config, t0)

    # ── Assembly ─────────────────────────────────────────────────────────

    def _assemble_and_save(
        self,
        features: pd.DataFrame,
        labels:   pd.DataFrame,
        config:   DatasetConfig,
        t0:       float,
    ) -> DatasetResult:

        # ── Step 1: Sort by time ───────────────────────────────────────
        features = features.sort_index()
        labels   = labels.sort_index()

        logger.info(
            "Loaded: features=%s  labels=%s",
            features.shape, labels.shape,
        )

        # ── Step 2: Inner join on timestamp index ─────────────────────
        dataset = features.join(labels, how="inner")
        feat_cols = [c for c in dataset.columns if c in features.columns]
        lbl_cols  = [c for c in dataset.columns if c in labels.columns]

        logger.info("After inner join: %d rows, %d features, %d labels",
                    len(dataset), len(feat_cols), len(lbl_cols))

        # ── Step 3: Drop rows with NaN in the primary target ──────────
        rows_before = len(dataset)
        if config.drop_na_labels and lbl_cols:
            target = config.primary_target
            if target and target in dataset.columns:
                dataset = dataset.dropna(subset=[target])
            else:
                # Drop rows where ALL label columns are NaN
                dataset = dataset.dropna(subset=lbl_cols, how="all")
            rows_dropped = rows_before - len(dataset)
            if rows_dropped > 0:
                logger.info("Dropped %d rows with NaN labels (%d remaining)",
                            rows_dropped, len(dataset))

        # ── Step 4: Validate ──────────────────────────────────────────
        val_report: Optional[DatasetValidationReport] = None
        if config.validate:
            vcfg = DatasetValidatorConfig(
                min_rows=config.min_rows,
                expected_columns=[],
            )
            val = DatasetValidator(vcfg)
            val_report = val.validate(
                dataset, feat_cols, lbl_cols,
                primary_target=config.primary_target,
            )

        # ── Step 5: Save ──────────────────────────────────────────────
        output_dir = Path(config.output_dir or self.output_dir) / config.symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        version     = self._next_version(output_dir, config.dataset_name, config.symbol)
        stem        = f"{config.dataset_name}_{config.symbol}_v{version}"

        parquet_path: Optional[Path] = None
        csv_path:     Optional[Path] = None

        if "parquet" in config.output_formats:
            parquet_path = output_dir / f"{stem}.parquet"
            dataset.to_parquet(parquet_path, index=True)
            logger.info("Dataset saved → %s", parquet_path)

        if "csv" in config.output_formats:
            csv_path = output_dir / f"{stem}.csv"
            dataset.to_csv(csv_path, index=True)
            logger.info("Dataset saved (CSV) → %s", csv_path)

        # ── Step 6: Metadata ──────────────────────────────────────────
        artefact_paths = {
            k: str(v) for k, v in {
                "parquet": parquet_path, "csv": csv_path,
            }.items() if v is not None
        }
        meta = DatasetMeta.build(
            dataset=dataset,
            feature_columns=feat_cols,
            label_columns=lbl_cols,
            symbol=config.symbol,
            dataset_name=config.dataset_name,
            dataset_version=version,
            label_version=config.label_version or 1,
            pipeline_version=config.pipeline_version,
            prediction_timeframe=config.prediction_timeframe,
            higher_timeframes=config.higher_timeframes,
            feature_set=config.feature_set,
            label_groups=config.label_groups or [],
            validation_passed=val_report.passed if val_report else True,
            validation_summary=str(val_report) if val_report else "",
            artefact_paths=artefact_paths,
        )

        meta_path = output_dir / f"{stem}_metadata.json"
        meta.to_json(meta_path)
        artefact_paths["metadata"] = str(meta_path)

        # ── Step 7: Reports ───────────────────────────────────────────
        report_paths = self.reporter.generate_all(dataset, meta)

        elapsed = time.perf_counter() - t0
        logger.info(
            "DatasetBuilder: DONE — %d rows × %d cols in %.2fs",
            len(dataset), len(dataset.columns), elapsed,
        )

        return DatasetResult(
            symbol=config.symbol,
            dataset=dataset,
            feature_columns=feat_cols,
            label_columns=lbl_cols,
            n_rows=len(dataset),
            n_features=len(feat_cols),
            n_labels=len(lbl_cols),
            parquet_path=parquet_path,
            csv_path=csv_path,
            metadata=meta,
            validation=val_report,
            report_paths=report_paths,
            build_time_s=elapsed,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _next_version(output_dir: Path, dataset_name: str, symbol: str) -> int:
        """Return the next unused version number."""
        import re
        pattern = re.compile(
            rf"{re.escape(dataset_name)}_{re.escape(symbol)}_v(\d+)\.parquet"
        )
        max_v = 0
        for f in output_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_v = max(max_v, int(m.group(1)))
        return max_v + 1
