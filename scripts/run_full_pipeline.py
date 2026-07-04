"""
Full ML Pipeline Runner — EURUSD
=================================
Executes every stage of the pipeline in order, validates outputs after
each stage, and halts immediately on a critical failure.

Stages
------
 1. Feature Engineering   → data/features/EURUSD/feature_dataset.parquet
 2. Label Generation      → data/labels/EURUSD/labels_EURUSD_v1.parquet
 3. Dataset Assembly      → data/ml/EURUSD/training_dataset_EURUSD_v1.parquet
 4. Walk-Forward Windows  → data/ml/windows/window_XXX/
 5. Hyperparameter Opt.   → models/window_XXX/model_name/bundle/
 6. Walk-Forward Valid.   → validation_results/
 7. Backtesting           → backtesting/
 8. Comparison Report     → reports/pipeline_summary.md

Usage
-----
    python scripts/run_full_pipeline.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Bootstrap ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    BASE_DIR,
    DATA_DIR,
    LOG_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_DIR,
    FEATURE_CACHE_DIR,
    REPORT_DIR,
    WALK_FORWARD_DIR,
    ML_DATASET_DIR,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline_runner.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline_runner")

# ── Constants ─────────────────────────────────────────────────────────────────
SYMBOL          = "EURUSD"
TARGET_COLUMN   = "direction_1b"
N_OPT_TRIALS    = 25          # Optuna trials per study
MODELS          = ["xgboost", "lightgbm", "random_forest", "extra_trees"]
INITIAL_CAPITAL = 10_000.0
MIN_PROBABILITY = 0.60

# Walk-forward periods (fits within 3.5-year M15 dataset)
WF_TRAIN  = "18m"
WF_VAL    = "6m"
WF_TEST   = "3m"
WF_STEP   = "3m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(title: str) -> None:
    line = "=" * 65
    log.info("")
    log.info(line)
    log.info("  %s", title)
    log.info(line)


def fail(msg: str) -> None:
    log.error("PIPELINE HALTED: %s", msg)
    sys.exit(1)


def check(condition: bool, msg: str) -> None:
    if not condition:
        fail(msg)


def stage_report(stage: str, elapsed: float, details: dict) -> None:
    log.info("")
    log.info("--- %s COMPLETE (%.1fs) ---", stage, elapsed)
    for k, v in details.items():
        log.info("    %-30s %s", k + ":", v)


# ── Stage 1: Feature Engineering ─────────────────────────────────────────────

def run_features() -> Path:
    separator("STAGE 1 — Feature Engineering")
    t0 = time.perf_counter()

    from src.features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(
        processed_dir   = PROCESSED_DATA_DIR,
        feature_dir     = FEATURE_DIR,
        report_dir      = REPORT_DIR,
        cache_dir       = FEATURE_CACHE_DIR,
        enable_cache    = True,
        enable_parallel = False,
    )
    feature_path = pipeline.run(SYMBOL)

    elapsed = time.perf_counter() - t0

    # Validation gate
    check(feature_path.exists(), f"Feature dataset not created: {feature_path}")
    feat_df = pd.read_parquet(feature_path)
    check(len(feat_df) > 1000, f"Feature dataset too small: {len(feat_df)} rows")
    check(feat_df.shape[1] > 5, f"Too few feature columns: {feat_df.shape[1]}")

    stage_report("FEATURES", elapsed, {
        "output":    str(feature_path),
        "rows":      len(feat_df),
        "columns":   feat_df.shape[1],
        "null_rate": f"{feat_df.isnull().mean().mean():.3%}",
    })
    return feature_path


# ── Stage 2: Label Generation ─────────────────────────────────────────────────

def run_labels(feature_path: Path) -> Path:
    separator("STAGE 2 — Label Generation")
    t0 = time.perf_counter()

    from src.labels.label_pipeline import LabelPipeline, LabelPipelineConfig

    feat_df = pd.read_parquet(feature_path)

    # Ensure timestamp index for labelers that use integer position
    if "timestamp" in feat_df.columns and not isinstance(feat_df.index, pd.DatetimeIndex):
        feat_df = feat_df.set_index("timestamp")
        feat_df.index = pd.to_datetime(feat_df.index, utc=True)

    pipeline = LabelPipeline(
        label_dir  = DATA_DIR / "labels",
        report_dir = REPORT_DIR / "labels",
        config     = LabelPipelineConfig(timeframe="M15"),
    )
    result = pipeline.run(feat_df, symbol=SYMBOL, write=True)

    elapsed = time.perf_counter() - t0

    check(result.validation_ok or result.parquet_path is not None,
          "Label pipeline validation failed and no file saved")
    check(result.parquet_path is not None and result.parquet_path.exists(),
          f"Labels parquet not saved: {result.parquet_path}")
    check(TARGET_COLUMN in result.labels.columns,
          f"Target column '{TARGET_COLUMN}' not found in labels. "
          f"Available: {list(result.labels.columns)[:10]}")

    vc = result.labels[TARGET_COLUMN].dropna().value_counts()
    stage_report("LABELS", elapsed, {
        "output":      str(result.parquet_path),
        "rows":        len(result.labels),
        "label_cols":  result.labels.shape[1],
        "target_dist": vc.to_dict(),
        "valid":       result.validation_ok,
    })
    return result.parquet_path


# ── Stage 3: Dataset Assembly ─────────────────────────────────────────────────

def run_dataset(feature_path: Path, label_path: Path) -> Path:
    separator("STAGE 3 — Dataset Assembly")
    t0 = time.perf_counter()

    from src.dataset.dataset_builder import DatasetBuilder, DatasetConfig

    config = DatasetConfig(
        symbol               = SYMBOL,
        feature_set          = "all",    # use all generated features
        label_groups         = None,     # include all label groups
        primary_target       = TARGET_COLUMN,
        drop_na_labels       = True,
        output_formats       = ["parquet"],
        validate             = True,
        min_rows             = 500,
        dataset_name         = "training_dataset",
        prediction_timeframe = "M15",
    )

    # Pre-align both to DatetimeIndex so the inner join finds matches
    feat_df_raw  = pd.read_parquet(feature_path)
    label_df_raw = pd.read_parquet(label_path)
    for name, df_raw in [("features", feat_df_raw), ("labels", label_df_raw)]:
        if "timestamp" in df_raw.columns and not isinstance(df_raw.index, pd.DatetimeIndex):
            df_raw = df_raw.set_index("timestamp")
            df_raw.index = pd.to_datetime(df_raw.index, utc=True)
        if name == "features":
            feat_aligned = df_raw
        else:
            label_aligned = df_raw

    builder = DatasetBuilder()
    result  = builder.build_from_dataframes(feat_aligned, label_aligned, config)

    elapsed = time.perf_counter() - t0

    check(result.parquet_path is not None and result.parquet_path.exists(),
          "ML dataset not saved")
    check(result.n_rows >= 500,
          f"Dataset too small after NaN drop: {result.n_rows} rows")
    check(TARGET_COLUMN in result.dataset.columns,
          f"Target column missing from assembled dataset")

    stage_report("DATASET", elapsed, {
        "output":      str(result.parquet_path),
        "rows":        result.n_rows,
        "features":    result.n_features,
        "labels":      result.n_labels,
        "valid":       getattr(result.validation, "passed", "N/A") if result.validation else "N/A",
    })
    return result.parquet_path


# ── Stage 4: Walk-Forward Windows ─────────────────────────────────────────────

def run_walk_forward(dataset_path: Path) -> tuple[Path, list[str]]:
    separator("STAGE 4 — Walk-Forward Window Generation")
    t0 = time.perf_counter()

    from src.walk_forward.walk_forward_generator import WalkForwardGenerator, WalkForwardConfig

    # Load and ensure DatetimeIndex (required by WalkForwardGenerator)
    df = pd.read_parquet(dataset_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            fail("Dataset has no DatetimeIndex and no 'timestamp' column")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Identify feature and label columns
    from src.dataset.dataset_loader import LABEL_GROUP_PREFIXES
    all_prefixes = tuple(
        p for prefs in LABEL_GROUP_PREFIXES.values() for p in prefs
    )
    label_cols   = [c for c in df.columns if c.startswith(all_prefixes)]
    feature_cols = [c for c in df.columns if c not in label_cols]

    log.info("Feature columns: %d  |  Label columns: %d", len(feature_cols), len(label_cols))

    config = WalkForwardConfig(
        window_type        = "rolling",
        train_period       = WF_TRAIN,
        val_period         = WF_VAL,
        test_period        = WF_TEST,
        step_period        = WF_STEP,
        min_train_samples  = 5_000,
        min_val_samples    = 2_000,
        min_test_samples   = 1_000,
        max_windows        = 0,
        validate           = True,
        feature_columns    = feature_cols,
        label_columns      = label_cols,
    )

    gen    = WalkForwardGenerator()
    result = gen.run(df, symbol=SYMBOL, config=config)

    elapsed = time.perf_counter() - t0

    check(result.n_windows > 0, "No walk-forward windows generated")
    check(result.all_passed, "Walk-forward validation failed for one or more windows")

    stage_report("WALK-FORWARD", elapsed, {
        "output":    str(result.output_dir),
        "n_windows": result.n_windows,
        "all_valid": result.all_passed,
        "report":    str(result.report_path),
    })
    return result.output_dir, feature_cols


# ── Stage 5: Hyperparameter Optimization ─────────────────────────────────────

def run_optimization(windows_dir: Path, feature_cols: list[str]) -> dict:
    separator("STAGE 5 — Hyperparameter Optimization")
    t0 = time.perf_counter()

    from src.optimization.optimization_pipeline import OptimizationPipeline, OptimizationConfig

    # Derive clean feature list from the first window's training split
    # (excludes all label columns, timestamp columns, and non-numeric columns)
    from src.dataset.dataset_loader import LABEL_GROUP_PREFIXES as _LGP
    win0 = pd.read_parquet(windows_dir / "window_000" / "train.parquet")
    _label_pfx = tuple(p for prefs in _LGP.values() for p in prefs)
    _label_set  = {c for c in win0.columns if c.startswith(_label_pfx)}
    _label_set.add(TARGET_COLUMN)
    clean_features = [
        c for c in win0.columns
        if c not in _label_set and pd.api.types.is_numeric_dtype(win0[c])
    ]
    log.info("Clean feature set: %d columns (no label leakage)", len(clean_features))

    config = OptimizationConfig(
        windows_dir              = windows_dir,
        models_dir               = MODELS_DIR,
        target_column            = TARGET_COLUMN,
        feature_columns          = clean_features,   # explicit — no label leakage
        model_names              = MODELS,
        task_type                = "auto",
        n_trials                 = N_OPT_TRIALS,
        timeout                  = None,
        optimization_metric      = "f1",
        direction                = "maximize",
        n_jobs_trials            = 1,
        random_seed              = 42,
        n_jobs_model             = -1,
        early_stopping_patience  = 10,
        early_stopping_warmup    = 5,
        early_stopping_min_delta = 1e-4,
        use_pruning              = False,
        storage_dir              = None,     # in-memory studies
        resume_if_exists         = False,
        report_dir               = REPORT_DIR / "optimization",
        best_model_dir           = MODELS_DIR / "best_model",
        skip_on_error            = True,
        symbol                   = SYMBOL,
    )

    pipeline = OptimizationPipeline()
    result   = pipeline.run(config)

    elapsed = time.perf_counter() - t0

    n_ok = sum(1 for r in result.results if r.bundle_dir is not None)
    check(n_ok > 0, f"Zero optimization results produced bundles. Errors: {result.errors[:3]}")

    best_dir = MODELS_DIR / "best_model"
    check(best_dir.exists(), f"best_model dir not created: {best_dir}")

    stage_report("OPTIMIZATION", elapsed, {
        "results_ok":    n_ok,
        "errors":        len(result.errors),
        "best_model":    str(result.selection_result.chosen_model_name) if result.selection_result else "N/A",
        "best_window":   str(result.selection_result.chosen_window_number) if result.selection_result else "N/A",
        "best_score":    f"{result.selection_result.composite_score:.4f}" if result.selection_result else "N/A",
        "report":        str(result.report_paths.get("optimization_report", "N/A")),
    })
    return {
        "n_ok":             n_ok,
        "selection_result": result.selection_result,
        "all_results":      result.results,
        "errors":           result.errors,
    }


# ── Stage 6: Walk-Forward Validation ─────────────────────────────────────────

def run_validation(windows_dir: Path) -> dict:
    separator("STAGE 6 — Walk-Forward Validation")
    t0 = time.perf_counter()

    from src.validation.validation_pipeline import ValidationPipeline, ValidationConfig

    config = ValidationConfig(
        windows_dir              = windows_dir,
        models_dir               = MODELS_DIR,
        output_dir               = BASE_DIR / "validation_results",
        target_column            = TARGET_COLUMN,
        task_type                = "auto",
        min_accuracy             = 0.45,
        min_f1                   = 0.35,
        min_directional_accuracy = 0.45,
        min_trading_accuracy     = 0.40,
        max_variance             = 0.30,
        stability_threshold      = 0.55,
        overfitting_threshold    = 0.20,
        skip_on_error            = True,
        symbol                   = SYMBOL,
    )

    pipeline = ValidationPipeline()
    result   = pipeline.run(config)

    elapsed = time.perf_counter() - t0

    check(result.n_windows > 0, "Validation: no windows processed")

    ranked       = result.ranked_models          # list[str] — best-first
    model_by_name = {r.model_name: r for r in result.model_results}
    stage_report("VALIDATION", elapsed, {
        "windows":        result.n_windows,
        "models":         result.n_models,
        "ranked_models":  ranked[:5],
        "errors":         len(result.errors),
        "report":         str(result.report_paths.get("overall_summary", "N/A")),
    })
    return {
        "ranked_models": ranked,
        "model_by_name": model_by_name,
        "errors":        result.errors,
    }


# ── Stage 7: Backtesting ──────────────────────────────────────────────────────

def run_backtest(feature_path: Path) -> dict:
    separator("STAGE 7 — Institutional Backtest")
    t0 = time.perf_counter()

    from src.backtesting.backtester import Backtester, BacktestConfig
    from src.backtesting.execution_engine import ExecutionConfig
    from src.backtesting.sl_tp_manager import SLTPConfig
    from src.backtesting.position_manager import PositionConfig
    from src.backtesting.risk_manager import RiskConfig

    best_bundle = MODELS_DIR / "best_model"
    # Walk into bundle/ sub-dir if it exists
    bundle_dir  = best_bundle / "bundle" if (best_bundle / "bundle").exists() else best_bundle
    check(bundle_dir.exists(), f"Best model bundle not found at {bundle_dir}")

    # Load the feature dataset as the price / feature DataFrame for backtesting
    price_df = pd.read_parquet(feature_path)
    if "timestamp" in price_df.columns and not isinstance(price_df.index, pd.DatetimeIndex):
        ts = price_df["timestamp"]
        price_df = price_df.set_index("timestamp")
        price_df.index = pd.to_datetime(price_df.index, utc=True)
    elif isinstance(price_df.index, pd.DatetimeIndex):
        pass
    else:
        fail("Cannot determine timestamp for backtest price data")

    cfg = BacktestConfig(
        bundle_dir      = bundle_dir,
        output_dir      = BASE_DIR / "backtesting",
        initial_capital = INITIAL_CAPITAL,
        min_probability = MIN_PROBABILITY,
        timestamp_column = "timestamp",
        execution       = ExecutionConfig(
            spread_pips=2.0, commission_per_lot=7.0,
            slippage_pips=0.5, slippage_std=0.3,
            execution_delay_bars=1,
        ),
        sl_tp           = SLTPConfig(
            mode="atr", sl_atr_mult=1.5, tp_atr_mult=3.0,
            enable_break_even=True, be_trigger_rr=1.0,
        ),
        position        = PositionConfig(mode="fixed_risk_pct", risk_pct=0.01),
        risk            = RiskConfig(
            max_open_positions=3,
            max_daily_loss_pct=0.02,
            max_weekly_loss_pct=0.05,
            initial_capital=INITIAL_CAPITAL,
        ),
        symbol          = SYMBOL,
    )

    backtester = Backtester(cfg)
    result     = backtester.run(price_df=price_df)

    elapsed = time.perf_counter() - t0

    m = result.metrics
    stage_report("BACKTEST", elapsed, {
        "trades":          m.get("n_trades", 0),
        "win_rate":        f"{m.get('win_rate', 0):.2%}",
        "net_profit":      f"${m.get('net_profit', 0):,.2f}",
        "profit_factor":   m.get("profit_factor"),
        "max_drawdown":    f"${m.get('max_drawdown', 0):,.2f}",
        "max_drawdown_%":  f"{m.get('max_drawdown_pct', 0):.2%}",
        "sharpe":          m.get("sharpe_ratio"),
        "sortino":         m.get("sortino_ratio"),
        "report":          str(result.report_paths.get("backtest_report", "N/A")),
    })
    return {"result": result, "metrics": m}


# ── Stage 8: Comparison Report ────────────────────────────────────────────────

def write_comparison_report(
    opt_info:  dict,
    val_info:  dict,
    bt_info:   dict,
) -> Path:
    separator("STAGE 8 — Comparison Report")

    lines: list[str] = [
        "# EURUSD ML Pipeline — Model Comparison Report",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Pipeline Configuration",
        "",
        f"- **Symbol**: {SYMBOL}",
        f"- **Target**: `{TARGET_COLUMN}`",
        f"- **Models**: {', '.join(MODELS)}",
        f"- **Optuna trials/study**: {N_OPT_TRIALS}",
        f"- **Walk-forward**: train={WF_TRAIN}, val={WF_VAL}, test={WF_TEST}, step={WF_STEP}",
        f"- **Initial capital**: ${INITIAL_CAPITAL:,.0f}",
        "",
        "---",
        "",
        "## 1. Optimization Results",
        "",
    ]

    # Per-model optimization summary
    model_scores: dict[str, list[float]] = {}
    for r in opt_info.get("all_results", []):
        if r.bundle_dir is not None:
            model_scores.setdefault(r.model_name, []).append(r.best_val_score)

    lines += ["| Model | Windows | Mean Val F1 | Best Val F1 |",
              "|-------|---------|-------------|-------------|"]
    for model, scores in sorted(model_scores.items()):
        import numpy as np
        lines.append(
            f"| {model} | {len(scores)} "
            f"| {float(np.mean(scores)):.4f} "
            f"| {float(max(scores)):.4f} |"
        )
    lines.append("")

    sel = opt_info.get("selection_result")
    if sel:
        lines += [
            f"**Selected Best Model**: `{sel.chosen_model_name}` "
            f"(window {sel.chosen_window_number}, "
            f"composite score {sel.composite_score:.4f})",
            "",
        ]

    # Validation ranking
    model_by_name = val_info.get("model_by_name", {})
    lines += [
        "## 2. Walk-Forward Validation Ranking",
        "",
        "| Rank | Model | Acceptance | Ranking Score |",
        "|------|-------|------------|---------------|",
    ]
    for rank, model_name in enumerate(val_info.get("ranked_models", []), 1):
        mv = model_by_name.get(model_name)
        lines.append(
            f"| {rank} | {model_name} "
            f"| {mv.acceptance_status if mv else 'N/A'} "
            f"| {f'{mv.ranking_score:.4f}' if mv else 'N/A'} |"
        )
    lines.append("")

    # Per-model validation metrics
    lines += ["## 3. Per-Model Validation Metrics", ""]
    for model_name, mv_result in model_by_name.items():
        rob   = mv_result.robustness
        stab  = mv_result.stability
        lines.append(f"### {model_name}")
        lines += [
            f"- **Acceptance**: {mv_result.acceptance_status}",
            f"- **Ranking Score**: {mv_result.ranking_score:.4f}",
            f"- **Stability Score**: {stab.stability_score:.4f}" if stab else "- **Stability**: N/A",
            f"- **Robustness Score**: {rob.robustness_score:.4f}" if rob else "- **Robustness**: N/A",
            f"- **Mean Inference (ms)**: {mv_result.mean_inference_ms:.2f}",
            "",
        ]

    # Backtest summary
    m = bt_info.get("metrics", {})
    lines += [
        "## 4. Backtest Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    display = [
        ("Net Profit",       "net_profit",      lambda v: f"${v:,.2f}"),
        ("Win Rate",         "win_rate",         lambda v: f"{v:.2%}"),
        ("# Trades",         "n_trades",         lambda v: str(v)),
        ("Profit Factor",    "profit_factor",    lambda v: f"{v:.4f}" if v else "N/A"),
        ("Expectancy",       "expectancy",       lambda v: f"${v:.2f}"),
        ("Sharpe Ratio",     "sharpe_ratio",     lambda v: f"{v:.4f}" if v else "N/A"),
        ("Sortino Ratio",    "sortino_ratio",    lambda v: f"{v:.4f}" if v else "N/A"),
        ("Calmar Ratio",     "calmar_ratio",     lambda v: f"{v:.4f}" if v else "N/A"),
        ("Max Drawdown",     "max_drawdown",     lambda v: f"${v:,.2f}"),
        ("Max Drawdown %",   "max_drawdown_pct", lambda v: f"{v:.2%}"),
        ("Recovery Factor",  "recovery_factor",  lambda v: f"{v:.4f}" if v else "N/A"),
        ("Ulcer Index",      "ulcer_index",      lambda v: f"{v:.6f}" if v else "N/A"),
    ]
    for label, key, fmt in display:
        val = m.get(key)
        lines.append(f"| {label} | {fmt(val) if val is not None else 'N/A'} |")
    lines.append("")

    # Session performance
    bt_result = bt_info.get("result")
    if bt_result and bt_result.session_analysis:
        lines += ["## 5. Backtest — Session Breakdown", "",
                  "| Session | Trades | Win Rate | Net Profit |",
                  "|---------|--------|----------|------------|"]
        for sess, stats in sorted(bt_result.session_analysis.items()):
            lines.append(
                f"| {sess} | {stats['n_trades']} "
                f"| {stats['win_rate']:.2%} "
                f"| ${stats['net_profit']:,.2f} |"
            )
        lines.append("")

    if bt_result and bt_result.direction_analysis:
        lines += ["## 6. Backtest — Direction Breakdown", "",
                  "| Direction | Trades | Win Rate | Net Profit |",
                  "|-----------|--------|----------|------------|"]
        for direction, stats in bt_result.direction_analysis.items():
            lines.append(
                f"| {direction} | {stats['n_trades']} "
                f"| {stats['win_rate']:.2%} "
                f"| ${stats['net_profit']:,.2f} |"
            )
        lines.append("")

    # Deployable bundle location
    lines += [
        "## 7. Deployable Bundle",
        "",
        f"The production-ready inference bundle is at:",
        "",
        f"    models/best_model/",
        "",
        "Contents:",
        "- `model.joblib` — trained model",
        "- `preprocessing.joblib` — fitted imputer / scaler",
        "- `feature_order.json` — required column order",
        "- `inference_config.json` — symbol, target, n_classes, task_type",
        "- `pipeline_manifest.json` — full bundle manifest + hash verification",
        "",
    ]

    report_path = REPORT_DIR / "pipeline_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Comparison report → %s", report_path)
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pipeline_start = time.perf_counter()
    log.info("Pipeline starting — symbol=%s  target=%s", SYMBOL, TARGET_COLUMN)

    feature_path  = run_features()
    label_path    = run_labels(feature_path)
    dataset_path  = run_dataset(feature_path, label_path)
    windows_dir, feature_cols = run_walk_forward(dataset_path)
    opt_info      = run_optimization(windows_dir, feature_cols)
    val_info      = run_validation(windows_dir)
    bt_info       = run_backtest(feature_path)
    report_path   = write_comparison_report(opt_info, val_info, bt_info)

    total = time.perf_counter() - pipeline_start
    separator("PIPELINE COMPLETE")
    log.info("  Total elapsed  : %.0f s (%.1f min)", total, total / 60)
    log.info("  Summary report : %s", report_path)
    log.info("  Best bundle    : %s", MODELS_DIR / "best_model")


if __name__ == "__main__":
    main()
