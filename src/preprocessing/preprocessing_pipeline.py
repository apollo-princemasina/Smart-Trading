"""PreprocessingPipeline — orchestrates all preprocessing steps for one symbol.

Steps per timeframe
-------------------
1. Load raw Parquet from data/raw/{symbol}/{TF}/
2. Validate raw OHLCV (10 structural checks)
3. Clean OHLCV (dedup, sort, drop corrupt candles, coerce types, ensure UTC)
4. Validate market calendar (weekend candles, gap classification)
5. Save cleaned file to data/processed/{symbol}/{TF}/

Cross-timeframe steps (run after all individual TFs are processed)
------------------------------------------------------------------
6. Cross-TF consistency checks (M15->H1, H1->H4, H4->D1)
7. Multi-TF merge: attach H1/H4/D1/W1 context onto M15 base (no lookahead)
8. Save merged dataset to data/processed/{symbol}/merged/

Report
------
9. Generate reports/data_quality_report.md
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .validate_ohlcv import OHLCVValidator, ValidationResult
from .clean_ohlcv import OHLCVCleaner, CleaningReport
from .market_calendar import ForexCalendar, CalendarReport
from .validate_timeframes import TimeframeValidator, CrossTFResult
from .merge_timeframes import TimeframeMerger, MergeReport
from .quality_report import QualityReportGenerator

logger = logging.getLogger(__name__)

# Cross-TF pairs checked in dependency order
_CROSS_TF_PAIRS: list[tuple[str, str]] = [
    ("M15", "H1"),
    ("H1",  "H4"),
    ("H4",  "D1"),
]


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for a single Forex symbol.

    Parameters
    ----------
    raw_data_dir : Path
        Root of raw data, e.g. data/raw. Files expected at
        {raw_data_dir}/{symbol}/{TF}/{symbol}_{TF}_*.parquet
    processed_dir : Path
        Root of processed output, e.g. data/processed.
    report_dir : Path
        Directory for quality reports.
    timeframes : list[str]
        Timeframes to process, ordered from highest to lowest granularity
        for the cross-TF check (W1, D1, H4, H1, M15).
    base_tf : str
        The lowest-granularity timeframe used as the merge base (default M15).
    """

    def __init__(
        self,
        raw_data_dir:  Path,
        processed_dir: Path,
        report_dir:    Path,
        timeframes:    list[str] | None = None,
        base_tf:       str = "M15",
    ) -> None:
        self.raw_data_dir  = raw_data_dir
        self.processed_dir = processed_dir
        self.report_dir    = report_dir
        self.timeframes    = timeframes or ["W1", "D1", "H4", "H1", "M15"]
        self.base_tf       = base_tf

        self._validator  = OHLCVValidator()
        self._cleaner    = OHLCVCleaner()
        self._calendar   = ForexCalendar()
        self._tf_checker = TimeframeValidator()
        self._reporter   = QualityReportGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, symbol: str) -> Path:
        """
        Process all timeframes for `symbol` and write outputs.

        Returns the path of the generated quality report.
        """
        logger.info("=" * 60)
        logger.info("Preprocessing pipeline starting — %s", symbol)
        logger.info("Timeframes: %s", self.timeframes)
        logger.info("=" * 60)

        validation_results: dict[str, ValidationResult] = {}
        cleaning_reports:   dict[str, CleaningReport]   = {}
        calendar_reports:   dict[str, CalendarReport]   = {}
        cleaned_dfs:        dict[str, pd.DataFrame]     = {}

        # ── Step 1-4: Per-timeframe processing ────────────────────────
        for tf in self.timeframes:
            logger.info("")
            logger.info("-- %s %s --", symbol, tf)

            raw_df = self._load_raw(symbol, tf)
            if raw_df is None:
                logger.warning("  No raw file found for %s %s — skipping.", symbol, tf)
                continue

            # Validate raw
            val_result = self._validator.validate(raw_df, tf)
            validation_results[tf] = val_result
            self._log_validation(tf, val_result)

            # Clean
            cleaned_df, clean_report = self._cleaner.clean(raw_df, tf)
            cleaning_reports[tf] = clean_report
            self._log_cleaning(tf, clean_report)

            # Calendar
            cal_report = self._calendar.validate(cleaned_df, tf)
            calendar_reports[tf] = cal_report
            self._log_calendar(tf, cal_report)

            # Save processed per-TF file
            self._save_processed(cleaned_df, symbol, tf)
            cleaned_dfs[tf] = cleaned_df

        # ── Step 5: Cross-TF consistency ──────────────────────────────
        logger.info("")
        logger.info("-- Cross-timeframe consistency checks --")
        cross_tf_results: list[CrossTFResult] = []

        for lower_tf, higher_tf in _CROSS_TF_PAIRS:
            if lower_tf not in cleaned_dfs or higher_tf not in cleaned_dfs:
                logger.info(
                    "  Skipping %s->%s (one or both not available).",
                    lower_tf, higher_tf,
                )
                continue

            result = self._tf_checker.validate(
                lower_df  = cleaned_dfs[lower_tf],
                higher_df = cleaned_dfs[higher_tf],
                lower_tf  = lower_tf,
                higher_tf = higher_tf,
            )
            cross_tf_results.append(result)
            self._log_cross_tf(result)

        # ── Step 6: Multi-TF merge (base = M15) ───────────────────────
        merge_report: MergeReport | None = None

        if self.base_tf in cleaned_dfs:
            logger.info("")
            logger.info("-- Multi-timeframe merge (base: %s) --", self.base_tf)

            higher_tfs = [tf for tf in self.timeframes if tf != self.base_tf]
            merger = TimeframeMerger(
                base_tf    = self.base_tf,
                higher_tfs = higher_tfs,
            )
            htf_dfs = {tf: cleaned_dfs[tf] for tf in higher_tfs if tf in cleaned_dfs}

            merged_df, merge_report = merger.merge(
                base_df = cleaned_dfs[self.base_tf],
                htf_dfs = htf_dfs,
            )
            self._log_merge(merge_report)
            self._save_merged(merged_df, symbol)
        else:
            logger.warning(
                "Base timeframe %s not available — skipping multi-TF merge.",
                self.base_tf,
            )

        # ── Step 7: Quality report ─────────────────────────────────────
        logger.info("")
        logger.info("-- Generating quality report --")
        report_path = self.report_dir / "data_quality_report.md"
        self._reporter.generate(
            symbol              = symbol,
            validation_results  = validation_results,
            cleaning_reports    = cleaning_reports,
            calendar_reports    = calendar_reports,
            cross_tf_results    = cross_tf_results,
            merge_report        = merge_report,
            output_path         = report_path,
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("Pipeline complete.")
        logger.info("  Processed files : %s/%s/", self.processed_dir, symbol)
        logger.info("  Quality report  : %s", report_path)
        logger.info("=" * 60)

        return report_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw(self, symbol: str, tf: str) -> pd.DataFrame | None:
        tf_dir = self.raw_data_dir / symbol / tf
        if not tf_dir.exists():
            return None

        files = sorted(tf_dir.glob("*.parquet"))
        if not files:
            return None

        frames = [pd.read_parquet(f) for f in files]
        df = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        logger.info("  Loaded %d rows from %d file(s).", len(df), len(files))
        return df

    def _save_processed(
        self, df: pd.DataFrame, symbol: str, tf: str
    ) -> Path:
        out_dir = self.processed_dir / symbol / tf
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{symbol}_{tf}_processed.parquet"
        df.to_parquet(path, index=False)
        mb = path.stat().st_size / 1_048_576
        logger.info("  Saved processed -> %s (%.1f MB, %d rows)", path.name, mb, len(df))
        return path

    def _save_merged(self, df: pd.DataFrame, symbol: str) -> Path:
        out_dir = self.processed_dir / symbol / "merged"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{symbol}_{self.base_tf}_merged.parquet"
        df.to_parquet(path, index=False)
        mb = path.stat().st_size / 1_048_576
        logger.info(
            "  Saved merged dataset -> %s (%.1f MB, %d rows, %d cols)",
            path.name, mb, len(df), df.shape[1],
        )
        return path

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_validation(tf: str, r: ValidationResult) -> None:
        status = "PASS" if r.passed else "FAIL"
        logger.info("  [validate] %s  total=%d  status=%s", tf, r.total_rows, status)
        for issue in r.issues:
            logger.error("    ERROR: %s", issue)
        for warn in r.warnings:
            logger.warning("    WARN:  %s", warn)

    @staticmethod
    def _log_cleaning(tf: str, r: CleaningReport) -> None:
        logger.info(
            "  [clean]    %s  in=%d out=%d removed=%d",
            tf, r.rows_input, r.rows_output, r.rows_removed,
        )
        for action in r.actions:
            logger.info("    -> %s", action)

    @staticmethod
    def _log_calendar(tf: str, r: CalendarReport) -> None:
        logger.info(
            "  [calendar] %s  weekend=%d expected_gaps=%d unexpected=%d",
            tf, r.weekend_candles, r.expected_gaps, r.unexpected_gaps,
        )
        for warn in r.warnings:
            logger.warning("    WARN: %s", warn)

    @staticmethod
    def _log_cross_tf(r: CrossTFResult) -> None:
        status = "PASS" if r.consistent else "FAIL"
        logger.info(
            "  [cross-tf] %s->%s  periods=%d  status=%s",
            r.lower_tf, r.higher_tf, r.periods_compared, status,
        )
        for issue in r.issues:
            logger.error("    ERROR: %s", issue)
        for warn in r.warnings:
            logger.warning("    WARN:  %s", warn)

    @staticmethod
    def _log_merge(r: MergeReport) -> None:
        logger.info(
            "  [merge]    base=%s  htf_count=%d  rows=%d",
            r.base_tf, r.htf_count, r.merged_rows,
        )
        for warn in r.warnings:
            logger.warning("    WARN: %s", warn)
