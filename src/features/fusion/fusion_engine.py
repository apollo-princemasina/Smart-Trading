"""End-to-end multi-timeframe fusion orchestrator."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd

from .feature_fusion   import FeatureFusion
from .fusion_validator import ValidationResult
from .timeframe_mapper import TimeframeMapper

logger = logging.getLogger(__name__)

_FUSED_FILENAME = "feature_dataset_fused.parquet"
_CACHE_SUBDIR   = ".fusion_cache"


def _save(df: pd.DataFrame, path: Path) -> None:
    """Save *df* to Parquet, preserving the DatetimeIndex."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True)


def _load(path: Path) -> pd.DataFrame:
    """Load a Parquet file, restoring the saved index."""
    return pd.read_parquet(path, engine="pyarrow")


def _fingerprint(timeframe_dfs: dict[str, pd.DataFrame]) -> str:
    """12-char MD5 hash over frame shapes + first timestamps."""
    parts = []
    for tf in sorted(timeframe_dfs):
        df  = timeframe_dfs[tf]
        ts0 = str(df.index[0]) if len(df) else "empty"
        parts.append(f"{tf}:{len(df)}:{ts0}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]  # noqa: S324


class FusionEngine:
    """
    Orchestrates multi-timeframe feature fusion for one or more symbols.

    Responsibilities
    ----------------
    * Calls :class:`FeatureFusion` for each symbol.
    * Caches results as Parquet under ``{base_dir}/{symbol}/``.
    * Supports parallel multi-symbol fusion via :class:`ThreadPoolExecutor`.
    * Provides incremental update — appends only new M15 bars without
      recomputing the full history.

    Output path: ``{base_dir}/{symbol}/feature_dataset_fused.parquet``
    """

    def __init__(
        self,
        base_dir:    str | Path = "data/features",
        cache:       bool       = True,
        validate:    bool       = True,
        strict:      bool       = False,
        max_workers: int        = 4,
    ):
        self._base_dir    = Path(base_dir)
        self._cache       = cache
        self._max_workers = max_workers
        self._fusion      = FeatureFusion(validate=validate, strict=strict)

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _fused_path(self, symbol: str) -> Path:
        return self._base_dir / symbol / _FUSED_FILENAME

    def _cache_path(self, symbol: str, fp: str) -> Path:
        return self._base_dir / symbol / _CACHE_SUBDIR / f"{fp}.parquet"

    # ── Single-symbol fusion ──────────────────────────────────────────────────

    def run(
        self,
        symbol:        str,
        timeframe_dfs: dict[str, pd.DataFrame],
        save:          bool                                    = True,
        on_complete:   Callable[[pd.DataFrame], None] | None  = None,
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Fuse all timeframe DataFrames for *symbol*.

        Args:
            symbol:        Instrument identifier (e.g. ``"EURUSD"``).
            timeframe_dfs: ``{timeframe: DataFrame}`` with features pre-computed.
            save:          Persist the fused DataFrame to Parquet.
            on_complete:   Optional callback invoked with the fused DataFrame.

        Returns:
            ``(fused_df, validation_result)``
        """
        logger.info("Fusing %s — TFs: %s", symbol, sorted(timeframe_dfs))

        # Check fingerprint-based cache
        if self._cache:
            fp    = _fingerprint(timeframe_dfs)
            cpath = self._cache_path(symbol, fp)
            if cpath.exists():
                logger.info("Cache hit: %s (%s)", symbol, fp)
                return _load(cpath), ValidationResult()

        fused, val = self._fusion.fuse(timeframe_dfs)

        if save:
            out = self._fused_path(symbol)
            out.parent.mkdir(parents=True, exist_ok=True)
            _save(fused, out)
            logger.info("Saved → %s", out)

        if self._cache:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            _save(fused, cpath)

        if on_complete is not None:
            on_complete(fused)

        return fused, val

    # ── Multi-symbol parallel fusion ──────────────────────────────────────────

    def run_many(
        self,
        symbols_dfs: dict[str, dict[str, pd.DataFrame]],
        save:        bool = True,
        parallel:    bool = True,
    ) -> dict[str, tuple[pd.DataFrame, ValidationResult]]:
        """
        Fuse multiple symbols, optionally in parallel.

        Args:
            symbols_dfs:  ``{symbol: {timeframe: DataFrame}}``.
            save:         Persist each fused DataFrame.
            parallel:     Use :class:`ThreadPoolExecutor` for concurrency.

        Returns:
            ``{symbol: (fused_df, validation_result)}``.
        """
        if not parallel or len(symbols_dfs) == 1:
            return {s: self.run(s, dfs, save=save) for s, dfs in symbols_dfs.items()}

        results: dict[str, tuple[pd.DataFrame, ValidationResult]] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self.run, sym, dfs, save): sym
                for sym, dfs in symbols_dfs.items()
            }
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    results[sym] = fut.result()
                    logger.info("Completed: %s", sym)
                except Exception as exc:
                    logger.error("Fusion failed for %s: %s", sym, exc)
                    results[sym] = (pd.DataFrame(), ValidationResult())
        return results

    # ── Incremental update ────────────────────────────────────────────────────

    def update_incremental(
        self,
        symbol:            str,
        new_timeframe_dfs: dict[str, pd.DataFrame],
        save:              bool = True,
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Append new M15 bars to the existing fused dataset.

        Only bars with index > last cached timestamp are fused.  The caller
        must supply enough HTF history so that ``merge_asof`` fills correctly
        for the new M15 bars.

        Args:
            symbol:             Instrument identifier.
            new_timeframe_dfs:  ``{timeframe: DataFrame}`` — recent data window.
            save:               Persist the updated dataset.

        Returns:
            ``(updated_fused_df, validation_result)``
        """
        fused_path = self._fused_path(symbol)

        if not fused_path.exists():
            logger.info("No existing dataset for %s — running full fusion", symbol)
            return self.run(symbol, new_timeframe_dfs, save=save)

        existing = _load(fused_path)
        last_ts  = existing.index[-1]

        m15_key = TimeframeMapper.normalise("M15")
        new_m15 = new_timeframe_dfs.get(m15_key, pd.DataFrame())
        if new_m15.empty:
            return existing, ValidationResult()

        incremental = new_m15[new_m15.index > last_ts]
        if incremental.empty:
            logger.info("No new M15 bars for %s since %s", symbol, last_ts)
            return existing, ValidationResult()

        logger.info(
            "Incremental: %d new M15 bars for %s", len(incremental), symbol
        )

        # Fuse only the new slice (HTF DataFrames must cover this window)
        partial = dict(new_timeframe_dfs)
        partial[m15_key] = incremental
        new_fused, val = self._fusion.fuse(partial)

        combined = pd.concat([existing, new_fused])
        combined = (
            combined[~combined.index.duplicated(keep="last")]
            .sort_index()
        )

        if save:
            _save(combined, fused_path)
            logger.info(
                "Updated → %s (%d rows total)", fused_path, len(combined)
            )

        return combined, val

    # ── Inspection helpers ────────────────────────────────────────────────────

    def load(self, symbol: str) -> pd.DataFrame | None:
        """Load the fused dataset for *symbol*, or ``None`` if absent."""
        path = self._fused_path(symbol)
        return _load(path) if path.exists() else None

    def describe(self, symbol: str) -> dict:
        """Return metadata about the fused dataset for *symbol*."""
        df = self.load(symbol)
        if df is None:
            return {"symbol": symbol, "status": "not_found"}
        return {
            "symbol":  symbol,
            "n_rows":  len(df),
            "n_cols":  df.shape[1],
            "start":   str(df.index[0]),
            "end":     str(df.index[-1]),
            "columns": df.columns.tolist(),
            "path":    str(self._fused_path(symbol)),
        }
