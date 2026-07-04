"""
Optimization Pipeline
=====================
Top-level orchestrator: loads walk-forward windows, trains and optimises every
requested model on every window, saves per-window inference bundles, and
selects the globally best model.

Output layout::

    models/
      window_001/
        xgboost/bundle/        ← full inference bundle
          model.joblib
          preprocessing.joblib
          feature_order.json
          ...
          pipeline_manifest.json
      best_model/              ← copy of the best window's bundle
        model.joblib
        ...

    reports/optimization/
      optimization_report.md
      optuna_results.csv
      best_parameters.json
      optimization_history.csv
      model_comparison.csv
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .artifact_manager import ArtifactManager, BundleConfig, ColumnImputer
from .model_selector import ModelSelector
from .objective import SUPPORTED_METRICS, ObjectiveFunction
from .optimizer import Optimizer, OptimizerConfig
from .optimization_reports import generate_optimization_report
from .search_space import SUPPORTED_MODELS

_SKLEARN_MODELS = frozenset({"random_forest", "extra_trees"})

logger = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class OptimizationConfig:
    """Full configuration for an optimization pipeline run.

    Attributes:
        windows_dir:            Directory that contains window_NNN/ folders.
        models_dir:             Root directory for model outputs.
        target_column:          Name of the label column.
        feature_columns:        Optional explicit feature list (None = all non-target).
        model_names:            Models to optimize (subset of SUPPORTED_MODELS).
        task_type:              "auto", "classification", or "regression".
        n_trials:               Optuna n_trials per study.
        timeout:                Per-study wall-clock budget in seconds (None = unlimited).
        optimization_metric:    Metric to optimise (from SUPPORTED_METRICS).
        direction:              "maximize" or "minimize".
        n_jobs_trials:          Parallel Optuna trials (>1 needs SQLite storage).
        random_seed:            Global random seed.
        n_jobs_model:           Parallelism inside each model fit.
        early_stopping_patience: Early-stopping patience (trials).
        early_stopping_warmup:  Warmup before early stopping activates.
        early_stopping_min_delta: Minimum improvement to reset patience.
        use_pruning:            Enable Optuna MedianPruner.
        storage_dir:            SQLite directory (None = in-memory studies).
        resume_if_exists:       Resume a prior study if SQLite storage is present.
        report_dir:             Directory for report files.
        best_model_dir:         Where to write the canonical best-model bundle.
        skip_on_error:          Skip a window/model on error instead of raising.
        schema_version:         Embedded in bundle metadata.
        label_version:          Embedded in bundle metadata.
        symbol:                 Instrument symbol (for metadata only).
        log_path:               Optional file path for pipeline log.
        baseline_val_scores:    Dict model_name → baseline score for comparison.
    """
    windows_dir:              Path
    models_dir:               Path
    target_column:            str
    feature_columns:          Optional[list[str]] = None
    model_names:              list[str]           = field(
                                  default_factory=lambda: list(SUPPORTED_MODELS)
                              )
    task_type:                str                 = "auto"
    n_trials:                 int                 = 50
    timeout:                  Optional[float]     = None
    optimization_metric:      str                 = "f1"
    direction:                str                 = "maximize"
    n_jobs_trials:            int                 = 1
    random_seed:              int                 = 42
    n_jobs_model:             int                 = -1
    early_stopping_patience:  int                 = 20
    early_stopping_warmup:    int                 = 10
    early_stopping_min_delta: float               = 1e-4
    use_pruning:              bool                = False
    storage_dir:              Optional[Path]      = None
    resume_if_exists:         bool                = True
    report_dir:               Optional[Path]      = None
    best_model_dir:           Optional[Path]      = None
    skip_on_error:            bool                = True
    schema_version:           str                 = "1.0.0"
    label_version:            str                 = "1.0.0"
    symbol:                   str                 = ""
    log_path:                 Optional[Path]      = None
    baseline_val_scores:      Optional[dict]      = None


@dataclass
class WindowOptResult:
    """Outcome of optimizing one model on one walk-forward window."""
    model_name:         str
    window_number:      int
    task_type:          str
    best_val_score:     float
    best_params:        dict
    n_trials_completed: int
    optimization_time_s: float
    training_time_s:    float
    prediction_time_s:  float
    n_train:            int
    n_val:              int
    n_test:             int
    n_features:         int
    train_metrics:      dict
    val_metrics:        dict
    test_metrics:       dict
    optimization_metric: str
    trial_history:      list[dict] = field(default_factory=list)
    bundle_dir:         Optional[Path]  = None
    baseline_val_score: Optional[float] = None


@dataclass
class PipelineOptResult:
    """Outcome of a full optimization pipeline run."""
    results:          list[WindowOptResult]
    selection_result: Optional[object]  = None   # SelectionResult
    errors:           list[str]         = field(default_factory=list)
    report_paths:     dict              = field(default_factory=dict)
    total_time_s:     float             = 0.0


# ── Pipeline ──────────────────────────────────────────────────────────────────

class OptimizationPipeline:
    """End-to-end hyperparameter optimization pipeline."""

    def run(self, config: OptimizationConfig) -> PipelineOptResult:
        """Run the full optimization pipeline.

        Steps:
          1. Discover walk-forward windows.
          2. For each window × each model: optimize → fit best params → save bundle.
          3. Select globally best model/window combination.
          4. Copy best bundle to config.best_model_dir.
          5. Generate all reports.

        Returns:
            PipelineOptResult with all outcomes.
        """
        _setup_logging(config.log_path)
        t_start   = time.monotonic()
        results   = []
        errors    = []
        windows   = self._discover_windows(config.windows_dir)

        if not windows:
            logger.warning("No walk-forward windows found in %s", config.windows_dir)

        task_type = config.task_type

        for win_dir in windows:
            wnum = _parse_window_number(win_dir)
            try:
                train_df, val_df, test_df = self._load_splits(win_dir, config.target_column)
            except Exception as exc:
                msg = f"Window {wnum}: failed to load splits — {exc}"
                logger.error(msg)
                errors.append(msg)
                continue

            # Detect task type once per window (it's the same for all models)
            if task_type == "auto":
                task_type = _detect_task_type(train_df, config.target_column)

            feat_cols = self._get_feature_columns(train_df, config)

            for model_name in config.model_names:
                try:
                    result = self._optimize_model(
                        model_name, wnum, task_type,
                        train_df, val_df, test_df, feat_cols, config,
                    )
                    results.append(result)
                except Exception as exc:
                    msg = f"Window {wnum} / {model_name}: {exc}"
                    logger.error(msg, exc_info=True)
                    errors.append(msg)
                    if not config.skip_on_error:
                        raise

        # Select best
        selector         = ModelSelector()
        selection_result = None
        if results:
            best = selector.select_best(results, task_type)
            if best is not None:
                best_model_dir = (
                    Path(config.best_model_dir)
                    if config.best_model_dir
                    else Path(config.models_dir) / "best_model"
                )
                selection_result = selector.create_best_bundle(best, best_model_dir)

        # Reports
        pipeline_result = PipelineOptResult(
            results=results,
            selection_result=selection_result,
            errors=errors,
            total_time_s=time.monotonic() - t_start,
        )
        report_dir = (
            Path(config.report_dir)
            if config.report_dir
            else Path(config.models_dir).parent / "reports" / "optimization"
        )
        try:
            paths = generate_optimization_report(pipeline_result, report_dir, task_type)
            pipeline_result.report_paths = {k: str(v) for k, v in paths.items()}
        except Exception as exc:
            logger.error("Report generation failed: %s", exc)

        logger.info(
            "Optimization pipeline done: %d results, %d errors, %.1fs",
            len(results), len(errors), pipeline_result.total_time_s,
        )
        return pipeline_result

    # ── Per-model optimization ─────────────────────────────────────────────────

    def _optimize_model(
        self,
        model_name:   str,
        window_number: int,
        task_type:    str,
        train_df:     pd.DataFrame,
        val_df:       pd.DataFrame,
        test_df:      pd.DataFrame,
        feat_cols:    list[str],
        config:       OptimizationConfig,
    ) -> WindowOptResult:
        target = config.target_column
        y_train = train_df[target].to_numpy()
        y_val   = val_df[target].to_numpy()
        y_test  = test_df[target].to_numpy()

        # Fit imputer on training data; gradient boosters don't need it
        imputer = ColumnImputer(
            apply_imputation=model_name in _SKLEARN_MODELS,
            strategy="median",
        )
        X_train_df = imputer.fit_transform(train_df[feat_cols])
        X_val_df   = imputer.transform(val_df[feat_cols])
        X_test_df  = imputer.transform(test_df[feat_cols])

        X_train = X_train_df.to_numpy(dtype=float, na_value=np.nan)
        X_val   = X_val_df.to_numpy(dtype=float, na_value=np.nan)
        X_test  = X_test_df.to_numpy(dtype=float, na_value=np.nan)

        # Optuna study
        opt_cfg = OptimizerConfig(
            n_trials                 = config.n_trials,
            timeout                  = config.timeout,
            direction                = config.direction,
            n_jobs_trials            = config.n_jobs_trials,
            random_seed              = config.random_seed,
            early_stopping_patience  = config.early_stopping_patience,
            early_stopping_warmup    = config.early_stopping_warmup,
            early_stopping_min_delta = config.early_stopping_min_delta,
            use_pruning              = config.use_pruning,
            storage_dir              = config.storage_dir,
            resume_if_exists         = config.resume_if_exists,
        )
        objective = ObjectiveFunction(
            model_name  = model_name,
            task_type   = task_type,
            X_train     = X_train,
            y_train     = y_train,
            X_val       = X_val,
            y_val       = y_val,
            metric      = config.optimization_metric,
            random_seed = config.random_seed,
            n_jobs      = config.n_jobs_model,
        )
        opt_result = Optimizer().optimize(
            objective     = objective,
            model_name    = model_name,
            metric        = config.optimization_metric,
            window_number = window_number,
            config        = opt_cfg,
        )

        # Retrain with best params on full train set
        from .search_space import get_search_space
        space      = get_search_space(model_name)
        best_model = space.build(
            opt_result.best_params, task_type,
            config.random_seed, config.n_jobs_model,
        )
        t_fit = time.monotonic()
        best_model.fit(X_train, y_train)
        training_time_s = time.monotonic() - t_fit

        t_pred = time.monotonic()
        _y_pred_val  = best_model.predict(X_val)
        _y_pred_test = best_model.predict(X_test)
        prediction_time_s = time.monotonic() - t_pred

        # Metrics
        from src.training.metrics import (
            compute_classification_metrics,
            compute_regression_metrics,
            compute_trading_metrics,
            detect_task_type,
        )
        y_prob_val  = best_model.predict_proba(X_val)  if hasattr(best_model, "predict_proba") else None
        y_prob_test = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
        y_prob_train = best_model.predict_proba(X_train) if hasattr(best_model, "predict_proba") else None
        y_pred_train = best_model.predict(X_train)

        if task_type == "classification":
            train_metrics = compute_classification_metrics(y_train, y_pred_train, y_prob_train)
            val_metrics   = compute_classification_metrics(y_val,   _y_pred_val,  y_prob_val)
            test_metrics  = compute_classification_metrics(y_test,  _y_pred_test, y_prob_test)
            trading_t     = compute_trading_metrics(y_train, y_pred_train, y_prob_train)
            trading_v     = compute_trading_metrics(y_val,   _y_pred_val,  y_prob_val)
            trading_x     = compute_trading_metrics(y_test,  _y_pred_test, y_prob_test)
            train_metrics.update(trading_t)
            val_metrics.update(trading_v)
            test_metrics.update(trading_x)
        else:
            train_metrics = compute_regression_metrics(y_train, y_pred_train)
            val_metrics   = compute_regression_metrics(y_val,   _y_pred_val)
            test_metrics  = compute_regression_metrics(y_test,  _y_pred_test)

        # Resolve window date boundaries from metadata.json if present
        win_meta = _load_window_meta(Path(config.windows_dir) / f"window_{window_number:03d}")

        # Save bundle
        bundle_cfg = BundleConfig(
            model_name           = model_name,
            task_type            = task_type,
            target_column        = target,
            feature_columns      = feat_cols,
            n_classes            = int(len(np.unique(y_train))) if task_type == "classification" else None,
            random_seed          = config.random_seed,
            schema_version       = config.schema_version,
            label_version        = config.label_version,
            window_number        = window_number,
            train_start          = win_meta.get("train_start"),
            train_end            = win_meta.get("train_end"),
            val_start            = win_meta.get("val_start"),
            val_end              = win_meta.get("val_end"),
            test_start           = win_meta.get("test_start"),
            test_end             = win_meta.get("test_end"),
            best_params          = opt_result.best_params,
            optimization_metric  = config.optimization_metric,
            n_trials             = opt_result.n_trials_completed,
            optimization_time_s  = opt_result.optimization_time_s,
            best_val_score       = opt_result.best_value,
            baseline_val_score   = (config.baseline_val_scores or {}).get(model_name),
            training_time_s      = training_time_s,
            prediction_time_s    = prediction_time_s,
            n_train_samples      = len(y_train),
            n_val_samples        = len(y_val),
            n_test_samples       = len(y_test),
            train_metrics        = train_metrics,
            val_metrics          = val_metrics,
            test_metrics         = test_metrics,
            study_name           = opt_result.study_name,
            symbol               = config.symbol,
        )
        bundle_dir = (
            Path(config.models_dir)
            / f"window_{window_number:03d}"
            / model_name
            / "bundle"
        )
        ArtifactManager.save_bundle(best_model, imputer, bundle_cfg, bundle_dir)

        return WindowOptResult(
            model_name          = model_name,
            window_number       = window_number,
            task_type           = task_type,
            best_val_score      = opt_result.best_value,
            best_params         = opt_result.best_params,
            n_trials_completed  = opt_result.n_trials_completed,
            optimization_time_s = opt_result.optimization_time_s,
            training_time_s     = training_time_s,
            prediction_time_s   = prediction_time_s,
            n_train             = len(y_train),
            n_val               = len(y_val),
            n_test              = len(y_test),
            n_features          = len(feat_cols),
            train_metrics       = train_metrics,
            val_metrics         = val_metrics,
            test_metrics        = test_metrics,
            optimization_metric = config.optimization_metric,
            trial_history       = opt_result.trial_history,
            bundle_dir          = bundle_dir,
            baseline_val_score  = (config.baseline_val_scores or {}).get(model_name),
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _discover_windows(windows_dir: Path) -> list[Path]:
        windows_dir = Path(windows_dir)
        if not windows_dir.exists():
            return []
        dirs = sorted(
            p for p in windows_dir.iterdir()
            if p.is_dir() and p.name.startswith("window_")
        )
        return dirs

    @staticmethod
    def _load_splits(
        win_dir: Path, target_column: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(win_dir / "train.parquet")
        val   = pd.read_parquet(win_dir / "validation.parquet")
        test  = pd.read_parquet(win_dir / "test.parquet")
        for name, df in [("train", train), ("validation", val), ("test", test)]:
            if target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found in {name} split "
                    f"of window {win_dir.name}."
                )
        return train, val, test

    @staticmethod
    def _get_feature_columns(df: pd.DataFrame, config: OptimizationConfig) -> list[str]:
        if config.feature_columns:
            # Filter provided list to only numeric columns present in df
            return [
                c for c in config.feature_columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
        # Auto-detect: all numeric columns except the target
        return [
            c for c in df.columns
            if c != config.target_column and pd.api.types.is_numeric_dtype(df[c])
        ]


def _parse_window_number(win_dir: Path) -> int:
    try:
        return int(win_dir.name.split("_")[-1])
    except ValueError:
        return 0


def _detect_task_type(df: pd.DataFrame, target_column: str) -> str:
    col = df[target_column].dropna()
    unique_vals = col.nunique()
    if unique_vals <= 20:
        # Integer/bool dtype → always classification
        if col.dtype.kind in ("i", "u", "b"):
            return "classification"
        # Float dtype where all values are whole numbers → label-encoded classes
        if col.dtype.kind == "f" and unique_vals > 0 and (col % 1 == 0).all():
            return "classification"
    return "regression"


def _load_window_meta(win_dir: Path) -> dict:
    meta_path = win_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    import json
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        # flatten nested structure if needed
        result = {}
        for split in ("train", "val", "validation", "test"):
            d = raw.get(split) or raw.get(f"{split}_stats") or {}
            start = d.get("start")
            end   = d.get("end")
            if start:
                result[f"{split}_start"] = str(start)
            if end:
                result[f"{split}_end"] = str(end)
        return result
    except Exception:
        return {}


def _setup_logging(log_path: Optional[Path]) -> None:
    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger("src.optimization").addHandler(handler)
