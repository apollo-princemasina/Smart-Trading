"""
Walk-Forward Generator
======================
Main orchestrator.  Loads an ML dataset, generates rolling window splits,
saves each split to Parquet, writes per-window ``metadata.json``, and
produces a summary Markdown report.

Output layout
-------------
    data/ml/windows/
        window_001/
            train.parquet
            validation.parquet
            test.parquet
            metadata.json
        window_002/
            ...
    reports/walk_forward/
        walk_forward_report.md

Usage
-----
    from src.walk_forward import WalkForwardGenerator, WalkForwardConfig

    gen = WalkForwardGenerator()
    result = gen.run(
        dataset_path="data/ml/EURUSD/training_dataset_EURUSD_v1.parquet",
        symbol="EURUSD",
        config=WalkForwardConfig(window_type="rolling", train_period="5y"),
    )
    print(f"Generated {result.n_windows} windows.")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .dataset_splitter import DatasetSplitter
from .reports import generate_walk_forward_report
from .split_validator import SplitValidator, SplitValidatorConfig
from .window_generator import WindowConfig, WindowGenerator
from .window_metadata import WindowMeta

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0.0"


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """All settings for one walk-forward generation run.

    Window-generation parameters are forwarded directly to ``WindowConfig``.
    """
    window_type:        str           = "rolling"
    train_period:       str           = "5y"
    val_period:         str           = "1y"
    test_period:        str           = "1y"
    step_period:        str           = "1y"
    anchor_date:        Optional[str] = None
    gap_bars:           int           = 0
    min_train_samples:  int           = 100
    min_val_samples:    int           = 50
    min_test_samples:   int           = 50
    max_windows:        int           = 0
    output_dir:         Optional[Path]= None    # defaults to data/ml/windows
    report_dir:         Optional[Path]= None    # defaults to reports/walk_forward
    validate:           bool          = True
    feature_columns:    list[str]     = field(default_factory=list)
    label_columns:      list[str]     = field(default_factory=list)
    schema_version:     str           = _SCHEMA_VERSION

    def to_window_config(self) -> WindowConfig:
        return WindowConfig(
            window_type       = self.window_type,
            train_period      = self.train_period,
            val_period        = self.val_period,
            test_period       = self.test_period,
            step_period       = self.step_period,
            anchor_date       = self.anchor_date,
            gap_bars          = self.gap_bars,
            min_train_samples = self.min_train_samples,
            min_val_samples   = self.min_val_samples,
            min_test_samples  = self.min_test_samples,
            max_windows       = self.max_windows,
        )

    def to_dict(self) -> dict:
        return {
            "window_type":       self.window_type,
            "train_period":      self.train_period,
            "val_period":        self.val_period,
            "test_period":       self.test_period,
            "step_period":       self.step_period,
            "anchor_date":       self.anchor_date,
            "gap_bars":          self.gap_bars,
            "min_train_samples": self.min_train_samples,
            "min_val_samples":   self.min_val_samples,
            "min_test_samples":  self.min_test_samples,
            "max_windows":       self.max_windows,
            "validate":          self.validate,
        }


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """Summary of a completed generation run."""
    n_windows:        int
    all_passed:       bool
    window_meta:      list[WindowMeta]
    output_dir:       Path
    report_path:      Path
    elapsed_seconds:  float

    def __str__(self) -> str:
        status = "ALL PASSED" if self.all_passed else "SOME FAILED"
        return (
            f"WalkForwardResult: {self.n_windows} windows | {status} | "
            f"{self.elapsed_seconds:.1f}s | {self.output_dir}"
        )


# ── Generator ─────────────────────────────────────────────────────────────────

class WalkForwardGenerator:
    """Orchestrates the full walk-forward dataset generation pipeline."""

    def __init__(self) -> None:
        self._gen       = WindowGenerator()
        self._splitter  = DatasetSplitter()

    # ------------------------------------------------------------------
    def run(
        self,
        dataset:      pd.DataFrame | Path | str,
        symbol:       str = "",
        config:       Optional[WalkForwardConfig] = None,
        feature_cols: Optional[list[str]] = None,
        label_cols:   Optional[list[str]] = None,
    ) -> WalkForwardResult:
        """Run walk-forward generation.

        Args:
            dataset:      Either a DataFrame or a path to a Parquet file.
            symbol:       Ticker symbol (used in directory names and reports).
            config:       Walk-forward configuration.  Uses defaults if None.
            feature_cols: Override for feature column list.
            label_cols:   Override for label column list.

        Returns:
            WalkForwardResult with paths and metadata.
        """
        t0  = time.monotonic()
        cfg = config or WalkForwardConfig()

        # ── Resolve feature/label columns ──────────────────────────────────
        if feature_cols is not None:
            cfg.feature_columns = feature_cols
        if label_cols is not None:
            cfg.label_columns = label_cols

        # ── Load dataset ───────────────────────────────────────────────────
        df = self._load_dataset(dataset)
        self._validate_dataframe(df)

        # ── Resolve column lists if not supplied ────────────────────────────
        if not cfg.feature_columns and not cfg.label_columns:
            cfg.feature_columns = list(df.columns)

        # ── Resolve output directories ──────────────────────────────────────
        from config.settings import ML_DATASET_DIR, REPORT_DIR
        base_windows = cfg.output_dir or (ML_DATASET_DIR / "windows")
        base_report  = cfg.report_dir or (REPORT_DIR / "walk_forward")

        # ── Generate window specs ───────────────────────────────────────────
        window_config = cfg.to_window_config()
        specs = self._gen.generate(df.index, window_config)
        logger.info("Generated %d window specs for symbol '%s'.", len(specs), symbol)

        if not specs:
            logger.warning("No windows generated — dataset may be too short.")
            result_dir = base_windows
            result_dir.mkdir(parents=True, exist_ok=True)
            rpt = generate_walk_forward_report([], cfg.to_dict(), base_report, symbol)
            return WalkForwardResult(
                n_windows=0, all_passed=True, window_meta=[],
                output_dir=result_dir, report_path=rpt,
                elapsed_seconds=time.monotonic() - t0,
            )

        # ── Split and save each window ──────────────────────────────────────
        validator = SplitValidator(SplitValidatorConfig(
            min_train_samples=cfg.min_train_samples,
            min_val_samples=cfg.min_val_samples,
            min_test_samples=cfg.min_test_samples,
        ))

        all_meta: list[WindowMeta] = []
        all_passed = True

        for spec in specs:
            split_result = self._splitter.split(df, spec)

            # Validate
            val_passed  = True
            val_issues: list[str] = []
            if cfg.validate:
                report = validator.validate(split_result)
                val_passed = report.passed
                val_issues = [f"[{i.severity}] {i.check}: {i.message}"
                              for i in report.failures() + report.warnings()]
                if not val_passed:
                    all_passed = False
                    logger.warning("Window %03d validation FAILED.", spec.window_number)

            # Save splits
            win_dir = base_windows / f"window_{spec.window_number:03d}"
            win_dir.mkdir(parents=True, exist_ok=True)
            artefacts = self._save_splits(split_result, win_dir)

            # Build and save metadata
            meta = WindowMeta.build(
                window_number     = spec.window_number,
                window_type       = cfg.window_type,
                train_df          = split_result.train,
                val_df            = split_result.validation,
                test_df           = split_result.test,
                train_period      = cfg.train_period,
                val_period        = cfg.val_period,
                test_period       = cfg.test_period,
                step_period       = cfg.step_period,
                gap_bars          = cfg.gap_bars,
                validation_passed = val_passed,
                validation_issues = val_issues,
                artefact_paths    = {k: str(v) for k, v in artefacts.items()},
                feature_cols      = cfg.feature_columns,
                label_cols        = cfg.label_columns,
                schema_version    = cfg.schema_version,
            )
            meta.to_json(win_dir / "metadata.json")
            all_meta.append(meta)
            logger.info("Window %03d saved → %s", spec.window_number, win_dir)

        # ── Summary report ──────────────────────────────────────────────────
        report_path = generate_walk_forward_report(
            all_meta, cfg.to_dict(), base_report, symbol,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "Walk-forward complete: %d windows in %.1fs → %s",
            len(all_meta), elapsed, base_windows,
        )

        return WalkForwardResult(
            n_windows       = len(all_meta),
            all_passed      = all_passed,
            window_meta     = all_meta,
            output_dir      = base_windows,
            report_path     = report_path,
            elapsed_seconds = elapsed,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_dataset(dataset: pd.DataFrame | Path | str) -> pd.DataFrame:
        if isinstance(dataset, pd.DataFrame):
            return dataset.copy()
        path = Path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise ValueError(f"Unsupported file type: {path.suffix!r}. Use .parquet or .csv.")

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Dataset is empty.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"Dataset must have a DatetimeIndex. Got: {type(df.index).__name__}."
            )
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "Dataset index is not monotonically increasing. Sort it before calling run()."
            )

    @staticmethod
    def _save_splits(
        split_result,
        win_dir: Path,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for name, df in [
            ("train",      split_result.train),
            ("validation", split_result.validation),
            ("test",       split_result.test),
        ]:
            p = win_dir / f"{name}.parquet"
            df.to_parquet(p, index=True)
            paths[name] = p
        return paths
