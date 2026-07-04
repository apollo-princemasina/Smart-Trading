"""FeaturePipeline — orchestrates the complete feature engineering workflow.

Pipeline steps
--------------
1.  Load the merged M15 OHLCV from ``data/processed/{symbol}/merged/``.
2.  Retrieve the execution order from ``FeatureRegistry`` (dependency-sorted).
3.  For each registered, enabled feature:
      a. Validate required input columns are present.
      b. Check the feature cache (if ``ENABLE_FEATURE_CACHE`` is True).
      c. Call ``feature.generate(df)`` with wall-clock timing.
      d. Run ``FeatureValidator.validate()`` on the output.
      e. Write to cache (if enabled).
      f. Collect the feature DataFrame.
4.  Merge the base OHLCV with all feature outputs into one wide DataFrame.
5.  Save to ``data/features/{symbol}/feature_dataset.parquet``.
6.  Generate ``reports/feature_pipeline_report.md``.

Design principles
-----------------
* The pipeline never modifies the raw or processed Parquet files.
* Feature generators are isolated — one generator's failure does not stop
  the pipeline; it logs the error and continues.
* Caching is keyed by a data fingerprint + feature name, so stale caches
  are automatically invalidated when the input data changes.
* All timing information is written to the report for performance tuning.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .feature_registry import FeatureRegistry
from .feature_validator import FeatureValidator, FeatureValidationReport, PipelineValidationSummary
from .feature_utils import (
    align_to_base,
    cache_path,
    check_required_columns,
    data_fingerprint,
    load_from_cache,
    load_parquet,
    merge_features,
    save_parquet,
    save_to_cache,
    timer,
)

if TYPE_CHECKING:
    from .base_feature import BaseFeature

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """End-to-end feature engineering pipeline for a single symbol.

    Parameters
    ----------
    processed_dir:
        Root of pre-processed data, e.g. ``data/processed``.
        The pipeline reads:
        ``{processed_dir}/{symbol}/merged/{symbol}_M15_merged.parquet``
    feature_dir:
        Root of feature output, e.g. ``data/features``.
        Writes: ``{feature_dir}/{symbol}/feature_dataset.parquet``
    report_dir:
        Report output directory.
        Writes: ``{report_dir}/feature_pipeline_report.md``
    cache_dir:
        Feature cache root, e.g. ``data/feature_cache``.
    enable_cache:
        Whether to read from / write to the feature cache.
    enable_parallel:
        Reserved for future use.  Currently, all features run sequentially.
    """

    def __init__(
        self,
        processed_dir:  Path,
        feature_dir:    Path,
        report_dir:     Path,
        cache_dir:      Path,
        enable_cache:   bool = True,
        enable_parallel: bool = False,
    ) -> None:
        self.processed_dir   = processed_dir
        self.feature_dir     = feature_dir
        self.report_dir      = report_dir
        self.cache_dir       = cache_dir
        self.enable_cache    = enable_cache
        self.enable_parallel = enable_parallel

        self._validator = FeatureValidator()

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, symbol: str) -> Path:
        """Execute the full pipeline for *symbol*.

        Returns
        -------
        Path
            Path to the generated feature dataset Parquet file.
        """
        run_start = datetime.now(timezone.utc)
        logger.info("=" * 65)
        logger.info("Feature engineering pipeline — %s", symbol)
        logger.info("=" * 65)

        # ── 1. Load merged input data ──────────────────────────────────────
        input_path = (
            self.processed_dir / symbol / "merged" /
            f"{symbol}_M15_merged.parquet"
        )
        logger.info("Loading merged dataset: %s", input_path.name)
        base_df = load_parquet(input_path)
        logger.info(
            "Input: %d rows x %d cols  [%s -> %s]",
            len(base_df),
            base_df.shape[1],
            base_df["timestamp"].iloc[0] if "timestamp" in base_df.columns else "?",
            base_df["timestamp"].iloc[-1] if "timestamp" in base_df.columns else "?",
        )

        # ── 2. Compute data fingerprint for cache keying ───────────────────
        fingerprint = data_fingerprint(base_df)
        logger.info("Data fingerprint: %s", fingerprint)

        # ── 3. Discover execution order ────────────────────────────────────
        execution_order = FeatureRegistry.get_execution_order()
        total = len(execution_order)
        logger.info("")
        logger.info("Registered features: %d enabled", total)
        for i, name in enumerate(execution_order, 1):
            fc = FeatureRegistry.get(name)
            logger.info(
                "  %2d. %-40s [%s]  deps=%s",
                i, name, fc.category, fc.dependencies or "none",
            )

        # ── 4. Execute each feature generator ─────────────────────────────
        logger.info("")
        logger.info("-- Executing feature generators --")

        feature_dfs: list[pd.DataFrame]             = []
        validation_reports: list[FeatureValidationReport] = []
        execution_metadata: list[dict]              = []

        # running_df accumulates base OHLCV + every feature's output so that
        # dependent features can read prior features' columns directly.
        running_df = base_df.copy()

        for idx, name in enumerate(execution_order, 1):
            logger.info("")
            logger.info("[%d/%d] %s", idx, total, name)

            feature_class = FeatureRegistry.get(name)
            feature       = feature_class()

            # Validate that required input columns are available (uses running_df
            # so dependent features can declare columns from earlier features).
            try:
                if feature.required_columns:
                    check_required_columns(running_df, feature.required_columns)
            except ValueError as exc:
                logger.error("  SKIP — missing required columns: %s", exc)
                execution_metadata.append({
                    "name": name, "status": "skipped",
                    "reason": str(exc), "ms": 0,
                })
                continue

            # Try cache first
            c_path    = cache_path(self.cache_dir, symbol, name, fingerprint)
            output_df = None

            if self.enable_cache:
                output_df = load_from_cache(c_path)
                if output_df is not None:
                    logger.info("  Cache hit (%.0f KB)", c_path.stat().st_size / 1024)

            # Generate if not cached — pass running_df so dependent features
            # have access to all previously computed columns.
            if output_df is None:
                try:
                    with timer(name) as elapsed:
                        output_df = feature.generate(running_df)
                    elapsed_ms = elapsed[0]
                    logger.info("  Generated in %.1f ms", elapsed_ms)

                    # Cache the result
                    if self.enable_cache and output_df is not None:
                        save_to_cache(output_df, c_path)

                except Exception as exc:
                    logger.exception("  ERROR in generate(): %s", exc)
                    execution_metadata.append({
                        "name": name, "status": "error",
                        "reason": str(exc), "ms": 0,
                    })
                    continue
                elapsed_ms = elapsed[0]
            else:
                elapsed_ms = 0.0

            if output_df is None:
                output_df = pd.DataFrame(index=base_df.index)

            # Align to base index (handles any off-by-one from rolling windows)
            output_df = align_to_base(output_df, base_df)

            # Accumulate new columns into running_df for downstream features.
            if not output_df.empty:
                new_cols = [c for c in output_df.columns if c not in running_df.columns]
                if new_cols:
                    running_df = pd.concat([running_df, output_df[new_cols]], axis=1)

            # Validate against running_df so column-shadow checks include
            # all previously generated columns (not just the base OHLCV).
            val_report = self._validator.validate(running_df, output_df, feature)
            validation_reports.append(val_report)

            status = "ok" if val_report.passed else "validation_failed"
            col_count = output_df.shape[1]
            logger.info(
                "  Validation: %s | cols: %d | NaN: %d | Inf: %d",
                status, col_count, val_report.nan_count, val_report.inf_count,
            )
            for issue in val_report.issues:
                logger.error("    ERROR: %s", issue)
            for warn in val_report.warnings:
                logger.warning("    WARN:  %s", warn)

            # Update feature metadata with execution time
            try:
                meta = feature.metadata()
                meta.execution_time_ms = elapsed_ms
                meta.output_columns    = list(output_df.columns)
            except Exception:
                pass  # Non-critical — metadata is for reporting only

            feature_dfs.append(output_df)
            execution_metadata.append({
                "name":       name,
                "category":   feature_class.category,
                "status":     status,
                "cols":       col_count,
                "nan_cells":  val_report.nan_count,
                "inf_cells":  val_report.inf_count,
                "ms":         round(elapsed_ms, 2),
            })

        # ── 5. Merge all features ──────────────────────────────────────────
        logger.info("")
        logger.info("-- Merging features --")
        try:
            dataset = merge_features(base_df, feature_dfs)
        except ValueError as exc:
            logger.error("Merge failed: %s", exc)
            logger.error("Saving base OHLCV without features.")
            dataset = base_df

        logger.info(
            "Final dataset: %d rows x %d cols",
            len(dataset), dataset.shape[1],
        )

        # ── 6. Save feature dataset ────────────────────────────────────────
        out_path = self.feature_dir / symbol / "feature_dataset.parquet"
        save_parquet(dataset, out_path)

        # ── 7. Validation summary ──────────────────────────────────────────
        val_summary = PipelineValidationSummary.from_reports(validation_reports)

        # ── 8. Generate report ─────────────────────────────────────────────
        run_end   = datetime.now(timezone.utc)
        total_ms  = (run_end - run_start).total_seconds() * 1000
        report_path = self._generate_report(
            symbol             = symbol,
            execution_order    = execution_order,
            execution_metadata = execution_metadata,
            val_summary        = val_summary,
            dataset            = dataset,
            run_start          = run_start,
            total_ms           = total_ms,
        )

        logger.info("")
        logger.info("=" * 65)
        logger.info("Pipeline complete in %.0f ms", total_ms)
        logger.info("  Feature dataset : %s", out_path)
        logger.info("  Pipeline report : %s", report_path)
        logger.info("=" * 65)

        return out_path

    # ── Report generation ────────────────────────────────────────────────────

    def _generate_report(
        self,
        symbol:             str,
        execution_order:    list[str],
        execution_metadata: list[dict],
        val_summary:        PipelineValidationSummary,
        dataset:            pd.DataFrame,
        run_start:          datetime,
        total_ms:           float,
    ) -> Path:
        report_path = self.report_dir / "feature_pipeline_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        ts = run_start.strftime("%Y-%m-%d %H:%M UTC")
        by_cat = FeatureRegistry.features_by_category()

        lines: list[str] = [
            f"# Feature Pipeline Report — {symbol}",
            "",
            f"Generated: {ts}  ",
            f"Total pipeline time: {total_ms:.0f} ms  ",
            f"Python: {sys.version.split()[0]}",
            "",
            "---",
            "",
            "## 1. Registered Features",
            "",
        ]

        # By-category table
        lines.append("| Category | Feature Name | Dependencies | Status |")
        lines.append("|---|---|---|---|")
        for cat in sorted(by_cat):
            for name in sorted(by_cat[cat]):
                fc   = FeatureRegistry.get(name)
                deps = ", ".join(fc.dependencies) if fc.dependencies else "—"
                dis  = name in FeatureRegistry._disabled
                status = "disabled" if dis else "enabled"
                lines.append(f"| {cat} | {name} | {deps} | {status} |")
        lines.append("")

        # Execution order
        lines += [
            "## 2. Execution Order (dependency-sorted)",
            "",
        ]
        for i, name in enumerate(execution_order, 1):
            fc = FeatureRegistry.get(name)
            lines.append(f"{i}. **{name}** `[{fc.category}]`")
        lines.append("")

        # Per-feature execution summary
        lines += [
            "## 3. Execution Summary",
            "",
            "| # | Feature | Category | Status | Columns | NaN | Inf | Time (ms) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, meta in enumerate(execution_metadata, 1):
            lines.append(
                f"| {i} "
                f"| {meta['name']} "
                f"| {meta.get('category', '—')} "
                f"| {meta['status']} "
                f"| {meta.get('cols', '—')} "
                f"| {meta.get('nan_cells', '—')} "
                f"| {meta.get('inf_cells', '—')} "
                f"| {meta['ms']} |"
            )
        lines.append("")

        # Validation summary
        ok_icon   = "**PASS**"
        fail_icon = "**FAIL**"
        overall   = ok_icon if val_summary.all_passed else fail_icon

        lines += [
            "## 4. Validation Summary",
            "",
            f"- Total features executed : {val_summary.total_features}",
            f"- Passed                  : {val_summary.passed_count}",
            f"- Failed                  : {val_summary.failed_count}",
            f"- Total NaN cells         : {val_summary.total_nan_cells:,}",
            f"- Total ±Inf cells        : {val_summary.total_inf_cells:,}",
            f"- Overall status          : {overall}",
            "",
        ]
        if val_summary.failed_features:
            lines.append("**Failed features:**")
            for name in val_summary.failed_features:
                lines.append(f"- {name}")
            lines.append("")

        # Output dataset info
        lines += [
            "## 5. Output Dataset",
            "",
            f"- Rows    : {len(dataset):,}",
            f"- Columns : {dataset.shape[1]}",
            "",
            "**Output columns:**",
            "",
        ]
        for col in dataset.columns:
            lines.append(f"- `{col}`")
        lines.append("")

        # Future modules roadmap
        lines += [
            "## 6. Future Feature Modules",
            "",
            "| Category | Planned Indicators |",
            "|---|---|",
            "| market_structure | BOS, CHoCH, MSS, Swing Highs/Lows, Order Blocks, FVGs |",
            "| liquidity        | Liquidity Pools, Equal Highs/Lows, Stop Hunt Detection |",
            "| sessions         | London/NY/Asia markers, Kill Zone flags, Session OHLC |",
            "| trend            | EMA Stack, Trend Bias, Higher-TF Direction |",
            "| volatility       | ATR, Bollinger Bands, Historical Volatility |",
            "| momentum         | RSI, MACD, Stochastic, ADX, Z-Score |",
            "| volume           | Delta Volume, Volume Profile, Cumulative Volume Delta |",
            "| labels           | Triple Barrier Labels, Binary Direction, RR Labels |",
            "",
        ]

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Feature pipeline report written -> %s", report_path)
        return report_path
