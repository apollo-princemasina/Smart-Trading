"""
Label Pipeline
==============
Orchestrates all five label generators into a single run.

Usage
-----
    from src.labels import LabelPipeline

    pipeline = LabelPipeline()
    result   = pipeline.run(ohlcv_df, symbol="EURUSD")
    # result.labels      — combined DataFrame (all label columns)
    # result.parquet_path — saved labels file

Or load from the Feature Store:
    result = pipeline.run_for_symbol("EURUSD")

The pipeline guarantees:
  - Input DataFrame is never modified (copy-on-entry)
  - Labels are saved to data/labels/{symbol}/ (never in the Feature Store)
  - Metadata + reports are generated automatically
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .market_bias    import MarketBiasLabeler,      MarketBiasConfig
from .trade_outcome  import TradeOutcomeLabeler,    TradeOutcomeConfig
from .setup_quality  import SetupQualityLabeler,    SetupQualityConfig
from .entry_timing   import EntryTimingLabeler,     EntryTimingConfig
from .trade_management import TradeManagementLabeler, TradeManagementConfig
from .label_validator  import LabelValidator,        LabelValidatorConfig
from .label_metadata   import LabelMeta
from .label_reports    import LabelReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class LabelPipelineConfig:
    market_bias:      MarketBiasConfig      = field(default_factory=MarketBiasConfig)
    trade_outcome:    TradeOutcomeConfig    = field(default_factory=TradeOutcomeConfig)
    setup_quality:    SetupQualityConfig    = field(default_factory=SetupQualityConfig)
    entry_timing:     EntryTimingConfig     = field(default_factory=EntryTimingConfig)
    trade_management: TradeManagementConfig = field(default_factory=TradeManagementConfig)
    validator:        LabelValidatorConfig  = field(default_factory=LabelValidatorConfig)
    label_version:    int   = 1
    timeframe:        str   = ""
    ohlcv_prefix:     str   = ""   # e.g. "h1_" if feature df uses prefixed columns


@dataclass
class LabelPipelineResult:
    symbol:         str
    labels:         pd.DataFrame
    metadata:       LabelMeta
    validation_ok:  bool
    parquet_path:   Optional[Path]
    report_paths:   dict[str, Path] = field(default_factory=dict)


class LabelPipeline:
    """End-to-end label generation for one symbol."""

    def __init__(
        self,
        label_dir:   Optional[Path] = None,
        report_dir:  Optional[Path] = None,
        config:      Optional[LabelPipelineConfig] = None,
    ) -> None:
        self.label_dir  = Path(label_dir  or Path("data") / "labels")
        self.report_dir = Path(report_dir or Path("reports") / "labels")
        self.config     = config or LabelPipelineConfig()

        cfg = self.config
        self._bias    = MarketBiasLabeler(cfg.market_bias)
        self._outcome = TradeOutcomeLabeler(cfg.trade_outcome)
        self._quality = SetupQualityLabeler(cfg.setup_quality)
        self._timing  = EntryTimingLabeler(cfg.entry_timing)
        self._mgmt    = TradeManagementLabeler(cfg.trade_management)
        self._val     = LabelValidator(cfg.validator)
        self._rep     = LabelReportGenerator(self.report_dir)

    # ------------------------------------------------------------------
    def run(
        self,
        df:          pd.DataFrame,
        symbol:      str,
        features_df: Optional[pd.DataFrame] = None,
        write:       bool = True,
    ) -> LabelPipelineResult:
        """
        Generate all labels from *df* (must contain OHLCV columns).

        Parameters
        ----------
        df          : OHLCV DataFrame (or feature DataFrame — prefix handled).
        symbol      : Instrument identifier (e.g. "EURUSD").
        features_df : Optional feature DataFrame for leakage validation.
        write       : If True, save labels parquet + reports to disk.
        """
        df = df.copy()   # read-only guarantee
        ohlcv = self._extract_ohlcv(df, self.config.ohlcv_prefix)

        logger.info("LabelPipeline.run: symbol=%s  rows=%d", symbol, len(ohlcv))

        # ── Run all labelers ─────────────────────────────────────────
        bias_res    = self._bias.fit(ohlcv)
        outcome_res = self._outcome.fit(ohlcv)
        quality_res = self._quality.fit(ohlcv)
        timing_res  = self._timing.fit(ohlcv)
        mgmt_res    = self._mgmt.fit(ohlcv)

        all_labels = pd.concat(
            [
                bias_res.labels,
                outcome_res.labels,
                quality_res.labels,
                timing_res.labels,
                mgmt_res.labels,
            ],
            axis=1,
        )

        # ── Validate ─────────────────────────────────────────────────
        val_report = self._val.validate(all_labels, features_df)
        logger.info("Validation: %s", val_report.summary)

        # ── Save ─────────────────────────────────────────────────────
        parquet_path: Optional[Path] = None
        if write:
            parquet_path = self._save_parquet(all_labels, symbol)

        # ── Metadata ─────────────────────────────────────────────────
        config_snapshot = {
            "market_bias":      asdict(self.config.market_bias),
            "trade_outcome":    asdict(self.config.trade_outcome),
            "setup_quality":    asdict(self.config.setup_quality),
            "entry_timing":     asdict(self.config.entry_timing),
            "trade_management": asdict(self.config.trade_management),
        }
        meta = LabelMeta.build(
            labels=all_labels,
            symbol=symbol,
            timeframe=self.config.timeframe,
            label_version=self.config.label_version,
            config_snapshot=config_snapshot,
            validation_summary=str(val_report),
            validation_passed=val_report.passed,
            artefact_paths={"labels_parquet": str(parquet_path or "")},
        )

        # ── Reports ──────────────────────────────────────────────────
        report_paths: dict[str, Path] = {}
        if write:
            report_paths = self._rep.generate_all(all_labels, meta)

        return LabelPipelineResult(
            symbol=symbol,
            labels=all_labels,
            metadata=meta,
            validation_ok=val_report.passed,
            parquet_path=parquet_path,
            report_paths=report_paths,
        )

    # ------------------------------------------------------------------
    def run_for_symbol(
        self,
        symbol:   str,
        version:  Optional[int] = None,
        write:    bool = True,
    ) -> LabelPipelineResult:
        """Load OHLCV from the Feature Store and run the pipeline."""
        try:
            from src.feature_store import FeatureStore
            from config.settings import FEATURE_STORE_DIR, SCHEMA_DIR

            fs = FeatureStore(FEATURE_STORE_DIR, SCHEMA_DIR)
            if version is not None:
                df = fs.load_version(symbol, version)
            else:
                df = fs.load_latest(symbol)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load symbol '{symbol}' from Feature Store: {exc}"
            ) from exc

        return self.run(df, symbol=symbol, write=write)

    def run_many(
        self,
        symbols: list[str],
        write:   bool = True,
    ) -> dict[str, LabelPipelineResult]:
        return {s: self.run_for_symbol(s, write=write) for s in symbols}

    # ------------------------------------------------------------------
    def _save_parquet(self, labels: pd.DataFrame, symbol: str) -> Path:
        sym_dir = self.label_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)

        # Versioned file — never overwrites
        version = 1
        while (sym_dir / f"labels_{symbol}_v{version}.parquet").exists():
            version += 1
        path = sym_dir / f"labels_{symbol}_v{version}.parquet"
        labels.to_parquet(path, index=True)
        logger.info("Labels saved → %s", path)
        return path

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Extract OHLCV columns.  Tries prefixed names first, then bare names.
        Raises ValueError if high/low/close are unavailable.
        """
        needed = ["open", "high", "low", "close", "volume"]
        rename = {}
        for col in needed:
            prefixed = f"{prefix}{col}"
            if prefixed in df.columns:
                rename[prefixed] = col
            elif col in df.columns:
                rename[col] = col   # identity

        mapped = {rename[k]: df[k] for k in rename}
        if not {"high", "low", "close"}.issubset(mapped.keys()):
            avail = [c for c in df.columns if any(
                c.endswith(n) for n in ("high", "low", "close", "open", "volume")
            )]
            raise ValueError(
                "Cannot find OHLCV columns. "
                f"Tried prefix='{prefix}'. Candidates: {avail[:10]}"
            )
        return pd.DataFrame(mapped, index=df.index)
