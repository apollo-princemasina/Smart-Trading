"""
FeatureQualityPipeline — end-to-end pipeline that integrates with the
Feature Store, runs all quality modules, generates reports, and returns
a :class:`FeatureQualityResults` object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_quality import FeatureQualityAnalyzer, FeatureQualityResults
from .feature_reports import FeatureReportGenerator

logger = logging.getLogger(__name__)


class FeatureQualityPipeline:
    """
    High-level interface: load data from the Feature Store, run analysis,
    write all reports, and return results.

    Parameters
    ----------
    feature_store:
        A ``FeatureStore`` instance (from ``src.feature_store``).
        If None, raw DataFrames must be passed to :meth:`run`.
    output_dir:
        Directory where reports are written (default ``reports/``).
    config:
        Optional configuration dict overriding defaults
        (see :attr:`FeatureQualityAnalyzer.DEFAULT_CONFIG`).
    pipeline_version:
        Version string injected into reports.
    """

    def __init__(
        self,
        feature_store   = None,
        output_dir:       str | Path = "reports",
        config:           dict[str, Any] | None = None,
        pipeline_version: str = "1.0.0",
    ):
        self._store    = feature_store
        self._out_dir  = Path(output_dir)
        self._config   = config or {}
        self._pip_ver  = pipeline_version

        self._analyser = FeatureQualityAnalyzer(
            feature_store=feature_store,
            config=config,
            pipeline_version=pipeline_version,
        )
        self._reporter = FeatureReportGenerator(output_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        df:         pd.DataFrame,
        symbol:     str,
        target:     pd.Series | None = None,
        target_col: str | None = None,
        write_reports: bool = True,
    ) -> FeatureQualityResults:
        """
        Run the full pipeline on *df*.

        Parameters
        ----------
        df:
            Feature DataFrame with a :class:`~pandas.DatetimeIndex`.
            This is **never modified**.
        symbol:
            Instrument identifier used in report headings.
        target:
            Optional target Series aligned with *df*.
        target_col:
            Column name in *df* to use as the target (extracted before analysis).
        write_reports:
            If True (default), write all report files to *output_dir*.

        Returns
        -------
        :class:`FeatureQualityResults`
        """
        logger.info(
            "FeatureQualityPipeline: starting analysis for %s (%d rows × %d cols)",
            symbol, len(df), df.shape[1],
        )

        results = self._analyser.run(
            df         = df,
            symbol     = symbol,
            target     = target,
            target_col = target_col,
        )

        if write_reports:
            paths = self._reporter.generate_all(results)
            logger.info("Reports written: %s", list(paths.values())[:5])

        return results

    def run_for_symbol(
        self,
        symbol:     str,
        target_col: str | None = None,
        version:    int | None = None,
        write_reports: bool = True,
    ) -> FeatureQualityResults:
        """
        Load the latest (or specific *version*) dataset for *symbol* from the
        Feature Store and run the full pipeline.

        Requires a Feature Store to have been provided at construction time.
        """
        if self._store is None:
            raise RuntimeError(
                "FeatureQualityPipeline requires a FeatureStore instance "
                "to call run_for_symbol(). Pass the feature_store argument."
            )

        if version is not None:
            df = self._store.load_version(symbol, version)
        else:
            df = self._store.load_latest(symbol)

        logger.info(
            "Loaded %s v%s from Feature Store (%d rows × %d cols)",
            symbol, version or "latest", len(df), df.shape[1],
        )

        return self.run(
            df            = df,
            symbol        = symbol,
            target_col    = target_col,
            write_reports = write_reports,
        )

    def run_many(
        self,
        symbols:    list[str],
        target_col: str | None = None,
        write_reports: bool = True,
    ) -> dict[str, FeatureQualityResults]:
        """Run the pipeline for multiple symbols."""
        all_results: dict[str, FeatureQualityResults] = {}
        for sym in symbols:
            try:
                all_results[sym] = self.run_for_symbol(
                    sym, target_col, write_reports=write_reports
                )
            except Exception as exc:
                logger.error("Pipeline failed for %s: %s", sym, exc)
        return all_results

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def analyser(self) -> FeatureQualityAnalyzer:
        return self._analyser

    @property
    def reporter(self) -> FeatureReportGenerator:
        return self._reporter
