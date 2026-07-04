"""
Training Pipeline
=================
Main orchestrator.  Discovers walk-forward windows, trains every requested
model on each window, saves artefacts, and produces all reports.

Usage
-----
    from src.training import TrainingPipeline, PipelineConfig

    cfg = PipelineConfig(
        windows_dir    = Path("data/ml/windows"),
        models_dir     = Path("models"),
        target_column  = "direction_1b",
        model_names    = ["xgboost", "lightgbm", "random_forest"],
    )
    result = TrainingPipeline().run(cfg)
    print(result)

Output artefacts
----------------
    models/window_001/xgboost.joblib
    models/window_001/xgboost_metadata.json
    ...
    reports/training/training_report.md
    reports/training/model_comparison.csv
    reports/training/metrics.csv
    reports/training/leaderboard.csv
    logs/training.log
"""
from __future__ import annotations

import logging
import logging.handlers
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .evaluation import Evaluator
from .metrics import detect_task_type
from .model_factory import SUPPORTED_MODELS, ModelFactory
from .model_registry import ModelRegistry
from .reports import (
    generate_comparison_csv,
    generate_leaderboard_csv,
    generate_metrics_csv,
    generate_training_report,
)
from .trainer import ModelWindowResult, Trainer, TrainerConfig

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All settings for one training pipeline run.

    Attributes:
        windows_dir:    Directory containing window_{NNN}/ sub-directories.
        models_dir:     Root directory for saved model files.
        target_column:  Name of the label column in the parquet files.
        feature_columns: Explicit feature list.  None = all non-target columns.
        model_names:    Models to train.  Defaults to all 5.
        task_type:      "auto" / "classification" / "regression".
        random_seed:    RNG seed for all models.
        n_jobs:         Parallelism passed to each model.
        report_dir:     Where to write Markdown / CSV reports.
        log_path:       Path for the training log file.
        schema_version: Schema version echoed into metadata JSON.
        symbol:         Ticker symbol (used in report headings).
        skip_on_error:  Continue past individual model failures (True) or raise (False).
    """
    windows_dir:     Path
    models_dir:      Path
    target_column:   str
    feature_columns: Optional[list[str]] = None
    model_names:     list[str]  = field(default_factory=lambda: list(SUPPORTED_MODELS))
    task_type:       str        = "auto"
    random_seed:     int        = 42
    n_jobs:          int        = -1
    report_dir:      Optional[Path] = None
    log_path:        Optional[Path] = None
    schema_version:  str        = "1.0.0"
    symbol:          str        = ""
    skip_on_error:   bool       = True

    def __post_init__(self) -> None:
        self.windows_dir = Path(self.windows_dir)
        self.models_dir  = Path(self.models_dir)
        if self.report_dir is None:
            from config.settings import BASE_DIR
            self.report_dir = BASE_DIR / "reports" / "training"
        if self.log_path is None:
            from config.settings import LOG_DIR
            self.log_path = LOG_DIR / "training.log"
        self.report_dir = Path(self.report_dir)
        self.log_path   = Path(self.log_path)

    def to_dict(self) -> dict:
        return {
            "windows_dir":    str(self.windows_dir),
            "models_dir":     str(self.models_dir),
            "target_column":  self.target_column,
            "model_names":    self.model_names,
            "task_type":      self.task_type,
            "random_seed":    self.random_seed,
            "n_jobs":         self.n_jobs,
            "schema_version": self.schema_version,
            "symbol":         self.symbol,
        }


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Summary of a completed training run."""
    n_windows:       int
    n_models:        int
    n_results:       int
    all_results:     list[ModelWindowResult]
    leaderboard:     pd.DataFrame
    report_path:     Path
    elapsed_seconds: float

    def __str__(self) -> str:
        status = "OK" if self.n_results == self.n_windows * self.n_models else "PARTIAL"
        return (
            f"PipelineResult [{status}]: "
            f"{self.n_windows} windows × {self.n_models} models = "
            f"{self.n_results}/{self.n_windows * self.n_models} runs | "
            f"{self.elapsed_seconds:.1f}s"
        )


# ── Pipeline ───────────────────────────────────────────────────────────────────

class TrainingPipeline:
    """Orchestrates baseline model training across all walk-forward windows."""

    def __init__(self) -> None:
        self._trainer  = Trainer()
        self._registry = ModelRegistry()
        self._evaluator = Evaluator()

    # ------------------------------------------------------------------
    def run(self, config: PipelineConfig) -> PipelineResult:
        """Execute the full training pipeline.

        Args:
            config: Pipeline configuration.

        Returns:
            PipelineResult with all results, leaderboard, and report paths.
        """
        t0 = time.monotonic()
        _configure_logging(config.log_path)

        config.report_dir.mkdir(parents=True, exist_ok=True)
        config.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "=== Baseline Training Pipeline started | symbol=%s | target=%s | models=%s ===",
            config.symbol, config.target_column, config.model_names,
        )

        # ── Discover windows ───────────────────────────────────────────────
        windows = _discover_windows(config.windows_dir)
        if not windows:
            logger.warning("No windows found in %s.", config.windows_dir)

        all_results: list[ModelWindowResult] = []

        for window_number, window_dir in windows:
            logger.info("── Window %03d (%s) ──", window_number, window_dir.name)

            # Load splits
            try:
                train_df = pd.read_parquet(window_dir / "train.parquet")
                val_df   = pd.read_parquet(window_dir / "validation.parquet")
                test_df  = pd.read_parquet(window_dir / "test.parquet")
            except Exception as exc:
                logger.error("Window %03d: failed to load parquet files: %s", window_number, exc)
                continue

            # Verify target exists in all splits
            missing = [
                name for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]
                if config.target_column not in df.columns
            ]
            if missing:
                logger.error(
                    "Window %03d: target '%s' missing from splits: %s. Skipping.",
                    window_number, config.target_column, missing,
                )
                continue

            # Auto-detect task type once per window
            y_sample = train_df[config.target_column].dropna()
            if y_sample.empty:
                logger.error("Window %03d: target column is entirely NaN. Skipping.", window_number)
                continue

            task_type = config.task_type
            if task_type == "auto":
                task_type = detect_task_type(y_sample)
            logger.info("  task_type = %s", task_type)

            trainer_cfg = TrainerConfig(
                target_column          = config.target_column,
                feature_columns        = config.feature_columns,
                task_type              = task_type,
                random_seed            = config.random_seed,
                n_jobs                 = config.n_jobs,
                compute_trading_metrics = (task_type == "classification"),
            )

            win_models_dir = config.models_dir / f"window_{window_number:03d}"
            win_models_dir.mkdir(parents=True, exist_ok=True)

            for model_name in config.model_names:
                try:
                    result = self._train_one(
                        model_name, task_type, train_df, val_df, test_df,
                        trainer_cfg, window_number, win_models_dir, config,
                    )
                    all_results.append(result)
                    _log_result(result)

                except Exception as exc:
                    logger.error(
                        "  [%s] FAILED: %s", model_name, exc,
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )
                    if not config.skip_on_error:
                        raise

        # ── Aggregate and report ───────────────────────────────────────────
        leaderboard = self._evaluator.build_leaderboard(all_results)

        report_path = generate_training_report(
            all_results, leaderboard, config.to_dict(),
            config.report_dir, config.symbol,
        )
        generate_metrics_csv(all_results, config.report_dir)
        generate_comparison_csv(all_results, config.report_dir)
        generate_leaderboard_csv(leaderboard, config.report_dir)

        elapsed = time.monotonic() - t0
        logger.info(
            "=== Pipeline complete: %d results in %.1fs ===",
            len(all_results), elapsed,
        )

        return PipelineResult(
            n_windows       = len(windows),
            n_models        = len(config.model_names),
            n_results       = len(all_results),
            all_results     = all_results,
            leaderboard     = leaderboard,
            report_path     = report_path,
            elapsed_seconds = elapsed,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _train_one(
        self,
        model_name:    str,
        task_type:     str,
        train_df:      pd.DataFrame,
        val_df:        pd.DataFrame,
        test_df:       pd.DataFrame,
        trainer_cfg:   TrainerConfig,
        window_number: int,
        win_models_dir: Path,
        config:        PipelineConfig,
    ) -> ModelWindowResult:
        model = ModelFactory.create(model_name, task_type,
                                    config.random_seed, config.n_jobs)
        result = self._trainer.train_window(
            model, model_name, train_df, val_df, test_df,
            trainer_cfg, window_number,
        )
        model_path = self._registry.save(
            model, result, win_models_dir,
            random_seed    = config.random_seed,
            schema_version = config.schema_version,
            train_start    = str(train_df.index[0])  if len(train_df) else None,
            train_end      = str(train_df.index[-1]) if len(train_df) else None,
            val_start      = str(val_df.index[0])    if len(val_df)   else None,
            val_end        = str(val_df.index[-1])   if len(val_df)   else None,
            test_start     = str(test_df.index[0])   if len(test_df)  else None,
            test_end       = str(test_df.index[-1])  if len(test_df)  else None,
        )
        result.model_path = model_path
        return result


# ── Utility functions ──────────────────────────────────────────────────────────

def _discover_windows(windows_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (window_number, Path) pairs."""
    windows: list[tuple[int, Path]] = []
    for d in sorted(Path(windows_dir).iterdir()):
        if d.is_dir() and d.name.startswith("window_"):
            try:
                n = int(d.name.split("_")[1])
                windows.append((n, d))
            except (IndexError, ValueError):
                continue
    return sorted(windows, key=lambda x: x[0])


def _configure_logging(log_path: Optional[Path]) -> None:
    """Add a file handler for training logs if not already present."""
    root = logging.getLogger()
    if log_path and not any(
        isinstance(h, logging.FileHandler) and
        getattr(h, "baseFilename", "") == str(log_path.resolve())
        for h in root.handlers
    ):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s"))
        root.addHandler(fh)


def _log_result(r: ModelWindowResult) -> None:
    """Emit a compact one-line summary of a training result."""
    if r.task_type == "classification":
        logger.info(
            "  ✓ %-15s  val_f1=%.4f  val_roc=%.4f  dir_acc=%.4f  fit=%.2fs",
            r.model_name,
            r.val_metrics.get("f1",       float("nan")),
            r.val_metrics.get("roc_auc",  float("nan")),
            r.val_metrics.get("directional_accuracy",
            r.val_metrics.get("accuracy", float("nan"))),
            r.training_time_seconds,
        )
    else:
        logger.info(
            "  ✓ %-15s  val_r2=%.4f  val_mae=%.4f  fit=%.2fs",
            r.model_name,
            r.val_metrics.get("r2",  float("nan")),
            r.val_metrics.get("mae", float("nan")),
            r.training_time_seconds,
        )
