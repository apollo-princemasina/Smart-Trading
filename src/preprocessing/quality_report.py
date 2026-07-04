"""Generate a Markdown data-quality report from preprocessing results."""

from __future__ import annotations

import logging
from pathlib import Path

from .validate_ohlcv import ValidationResult
from .clean_ohlcv import CleaningReport
from .market_calendar import CalendarReport
from .validate_timeframes import CrossTFResult
from .merge_timeframes import MergeReport

logger = logging.getLogger(__name__)


def _bool_icon(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{100 * n / total:.2f}%"


class QualityReportGenerator:
    """
    Accepts all preprocessing results and writes a comprehensive
    Markdown report to `reports/data_quality_report.md`.
    """

    def generate(
        self,
        symbol: str,
        validation_results:  dict[str, ValidationResult],
        cleaning_reports:    dict[str, CleaningReport],
        calendar_reports:    dict[str, CalendarReport],
        cross_tf_results:    list[CrossTFResult],
        merge_report:        MergeReport | None,
        output_path:         Path,
    ) -> Path:
        sections = [
            self._header(symbol),
            self._validation_section(validation_results),
            self._cleaning_section(cleaning_reports),
            self._calendar_section(calendar_reports),
            self._cross_tf_section(cross_tf_results),
            self._merge_section(merge_report),
            self._summary_section(
                validation_results, cleaning_reports, cross_tf_results
            ),
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n\n".join(s for s in sections if s)
        output_path.write_text(content, encoding="utf-8")
        logger.info("Quality report written -> %s", output_path)
        return output_path

    # ------------------------------------------------------------------

    @staticmethod
    def _header(symbol: str) -> str:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"# Data Quality Report — {symbol}\n\n"
            f"Generated: {ts}  \n"
            f"Source: MetaTrader 5 / MetaQuotes-Demo  \n"
            f"Pipeline: ICT + ML Smart Trading Preprocessing\n"
        )

    @staticmethod
    def _validation_section(results: dict[str, ValidationResult]) -> str:
        if not results:
            return ""

        lines = ["## 1. Raw OHLCV Validation\n"]
        lines.append(
            "| TF | Rows | Dupes | OHLC Err | Neg Price | "
            "Neg Vol | Large Spread | Const | Gaps | Status |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        for tf, r in results.items():
            status = "**PASS**" if r.passed else "**FAIL**"
            lines.append(
                f"| {tf} | {r.total_rows:,} "
                f"| {r.duplicate_timestamps} "
                f"| {r.ohlc_violations} "
                f"| {r.negative_prices} "
                f"| {r.negative_volumes} "
                f"| {r.large_spreads} "
                f"| {r.constant_candles} "
                f"| {r.unexpected_gaps} "
                f"| {status} |"
            )

        lines.append("")
        for tf, r in results.items():
            for issue in r.issues:
                lines.append(f"- **[{tf} ERROR]** {issue}")
            for warn in r.warnings:
                lines.append(f"- [{tf} WARN] {warn}")

        return "\n".join(lines)

    @staticmethod
    def _cleaning_section(reports: dict[str, CleaningReport]) -> str:
        if not reports:
            return ""

        lines = ["## 2. Cleaning Summary\n"]
        lines.append(
            "| TF | Input | Output | Removed | Dupes | Bad OHLC | "
            "Zero Price | Sorted | UTC |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for tf, r in reports.items():
            lines.append(
                f"| {tf} "
                f"| {r.rows_input:,} "
                f"| {r.rows_output:,} "
                f"| {r.rows_removed} "
                f"| {r.duplicates_removed} "
                f"| {r.invalid_ohlc_removed} "
                f"| {r.zero_price_removed} "
                f"| {'yes' if r.rows_sorted else 'no'} "
                f"| {'yes' if r.tz_coerced else 'already'} |"
            )

        lines.append("")
        for tf, r in reports.items():
            for action in r.actions:
                lines.append(f"- [{tf}] {action}")

        return "\n".join(lines)

    @staticmethod
    def _calendar_section(reports: dict[str, CalendarReport]) -> str:
        if not reports:
            return ""

        lines = ["## 3. Market Calendar Validation\n"]
        lines.append(
            "| TF | Weekend Candles | Expected Gaps | Unexpected Gaps | Thin-Market Rows |"
        )
        lines.append("|---|---|---|---|---|")

        for tf, r in reports.items():
            lines.append(
                f"| {tf} "
                f"| {r.weekend_candles} "
                f"| {r.expected_gaps} "
                f"| {r.unexpected_gaps} "
                f"| {r.thin_market_rows} |"
            )

        lines.append("")
        for tf, r in reports.items():
            for warn in r.warnings:
                lines.append(f"- [{tf}] {warn}")

        return "\n".join(lines)

    @staticmethod
    def _cross_tf_section(results: list[CrossTFResult]) -> str:
        if not results:
            return ""

        lines = ["## 4. Cross-Timeframe Consistency\n"]
        lines.append(
            "| Pair | Periods | Open Err | High Err | Low Err | Close Err | Vol Err | Incomplete | Status |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for r in results:
            pair   = f"{r.lower_tf}->{r.higher_tf}"
            status = "**PASS**" if r.consistent else "**FAIL**"
            lines.append(
                f"| {pair} "
                f"| {r.periods_compared:,} "
                f"| {r.open_mismatches} "
                f"| {r.high_mismatches} "
                f"| {r.low_mismatches} "
                f"| {r.close_mismatches} "
                f"| {r.volume_mismatches} "
                f"| {r.incomplete_periods} "
                f"| {status} |"
            )

        lines.append("")
        for r in results:
            for issue in r.issues:
                lines.append(f"- **[{r.lower_tf}->{r.higher_tf} ERROR]** {issue}")
            for warn in r.warnings:
                lines.append(f"- [{r.lower_tf}->{r.higher_tf} WARN] {warn}")

        return "\n".join(lines)

    @staticmethod
    def _merge_section(report: MergeReport | None) -> str:
        if report is None:
            return ""

        lines = ["## 5. Multi-Timeframe Merge\n"]
        lines.append(f"- Base timeframe: **{report.base_tf}**")
        lines.append(f"- Higher TFs attached: **{report.htf_count}**")
        lines.append(f"- Base rows: {report.base_rows:,}")
        lines.append(f"- Merged rows: {report.merged_rows:,}")

        if report.null_htf_rows:
            lines.append("\n**Rows without a completed HTF candle (warm-up period):**\n")
            for tf, n in report.null_htf_rows.items():
                lines.append(f"  - {tf}: {n} rows")

        if report.warnings:
            lines.append("")
            for w in report.warnings:
                lines.append(f"- [WARN] {w}")

        return "\n".join(lines)

    @staticmethod
    def _summary_section(
        validation_results: dict[str, ValidationResult],
        cleaning_reports:   dict[str, CleaningReport],
        cross_tf_results:   list[CrossTFResult],
    ) -> str:
        lines = ["## 6. Overall Assessment\n"]

        hard_fails = [
            tf for tf, r in validation_results.items() if not r.passed
        ]
        cross_fails = [
            f"{r.lower_tf}->{r.higher_tf}"
            for r in cross_tf_results
            if not r.consistent
        ]

        total_removed = sum(r.rows_removed for r in cleaning_reports.values())

        if not hard_fails and not cross_fails:
            lines.append(
                "**All checks passed.** Data is clean and consistent "
                "across all timeframes. Ready for feature engineering."
            )
        else:
            lines.append("**Issues requiring attention before feature engineering:**\n")
            for tf in hard_fails:
                lines.append(f"- Raw validation FAILED for {tf}")
            for pair in cross_fails:
                lines.append(f"- Cross-TF consistency FAILED for {pair}")

        lines.append(f"\nTotal rows cleaned (removed): **{total_removed:,}**")

        return "\n".join(lines)
