"""Build inference features for live data.

Bridge between raw per-timeframe OHLCV data (Twelve Data or MT5) and the
247-feature vector expected by InferencePipeline.predict().

Typical usage
-------------
    from src.inference.feature_builder import build_inference_features

    feature_df = build_inference_features(
        m15_df,
        htf_dfs={"H1": h1_df, "H4": h4_df, "D1": d1_df, "W1": w1_df},
    )
    # feature_df is ready for InferencePipeline.predict(feature_df)
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Two levels up from src/inference/ → project root
_BASE_DIR = Path(__file__).resolve().parents[2]

_DEFAULT_SYMBOL = "EURUSD"

# Spread constant for sources that omit FX spread (e.g. Twelve Data free tier).
# 15 = 1.5 pips in 5-digit broker notation (0.00015 price units) — matches
# median EURUSD spread observed during the 2022-2024 training period.
_SPREAD_FILL = 15

# Minimum columns required in m15_df before merging / feature generation
_REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "tick_volume"]


def build_inference_features(
    m15_df:    pd.DataFrame,
    htf_dfs:   Optional[dict] = None,
    *,
    symbol:    str = _DEFAULT_SYMBOL,
    spread_fill: int = _SPREAD_FILL,
) -> pd.DataFrame:
    """Merge multi-timeframe OHLCV data and produce the full feature dataset.

    Parameters
    ----------
    m15_df : pd.DataFrame
        M15 OHLCV bars in MT5/Twelve-Data schema.
        Required columns: timestamp (UTC-aware), open, high, low, close,
        tick_volume.
        Optional columns: spread (filled with *spread_fill* if absent or all-zero),
        real_volume (filled with 0 if absent).
        Rows must be sorted ascending by timestamp.
    htf_dfs : dict[str, pd.DataFrame], optional
        Higher-timeframe DataFrames keyed by timeframe string:
        ``{"H1": df, "H4": df, "D1": df, "W1": df}``.
        Each shares the same column schema as *m15_df*.
        If *None*, *m15_df* is assumed to already contain all prefixed HTF
        columns (``h1_open``, ``h4_close``, etc.) from a prior merge step.
    symbol : str
        Internal symbol name without separator (default ``"EURUSD"``).
    spread_fill : int
        Spread value injected when the ``spread`` column is absent or all-zero.
        Default 15 (1.5 pips for a 5-digit broker).

    Returns
    -------
    pd.DataFrame
        Feature dataset: one row per M15 bar, all engineered feature columns
        populated. First N rows may carry NaN values from rolling-window warmup
        — these are handled by XGBoost natively (no imputation needed).
        The returned DataFrame is ready for::

            InferencePipeline(bundle_dir).predict(feature_df)

    Raises
    ------
    ValueError
        If *m15_df* is missing required columns or has wrong timestamp dtype.
    RuntimeError
        If the feature pipeline produces no output file.
    """
    from src.preprocessing.merge_timeframes import TimeframeMerger
    from src.features.feature_pipeline     import FeaturePipeline
    from src.features.feature_utils        import save_parquet

    _validate(m15_df)

    # Inject spread / real_volume if the data source omits them
    m15_df = m15_df.copy()
    if "spread" not in m15_df.columns or (m15_df["spread"] == 0).all():
        m15_df["spread"] = spread_fill
        logger.debug("spread injected: %d", spread_fill)
    if "real_volume" not in m15_df.columns:
        m15_df["real_volume"] = 0

    tmp = Path(tempfile.mkdtemp(prefix="st_infer_"))
    try:
        # ── 1. Multi-timeframe merge ─────────────────────────────────────────────
        if htf_dfs is not None:
            merger = TimeframeMerger(
                base_tf    = "M15",
                higher_tfs = list(htf_dfs.keys()),
            )
            merged_df, report = merger.merge(m15_df, htf_dfs)
            for w in report.warnings:
                logger.warning("TimeframeMerger: %s", w)
            logger.debug(
                "Merged %d x %d  (base=%d, HTFs=%d merged)",
                report.merged_rows, merged_df.shape[1],
                report.base_rows, report.htf_count,
            )
        else:
            merged_df = m15_df

        # Drop auxiliary merger timestamp columns (e.g. h1_timestamp)
        aux_cols = [c for c in merged_df.columns
                    if c.endswith("_timestamp") and c != "timestamp"]
        if aux_cols:
            merged_df = merged_df.drop(columns=aux_cols)

        # ── 2. Write merged parquet to temp workspace ────────────────────────────
        merged_dir = tmp / "processed" / symbol / "merged"
        merged_dir.mkdir(parents=True)
        save_parquet(merged_df, merged_dir / f"{symbol}_M15_merged.parquet")

        # ── 3. Feature engineering pipeline ─────────────────────────────────────
        feat_dir   = tmp / "features"
        report_dir = tmp / "pipeline_reports"
        cache_dir  = tmp / "cache"
        for d in (feat_dir, report_dir, cache_dir):
            d.mkdir(parents=True, exist_ok=True)

        pipeline = FeaturePipeline(
            processed_dir   = tmp / "processed",
            feature_dir     = feat_dir,
            report_dir      = report_dir,
            cache_dir       = cache_dir,
            enable_cache    = False,   # no stale cache in live inference
            enable_parallel = False,
        )
        out_path = pipeline.run(symbol)

        if not out_path.exists():
            raise RuntimeError(
                f"FeaturePipeline produced no output for {symbol}. "
                "Check pipeline logs for generator errors."
            )

        # ── 4. Load, normalise, and return ───────────────────────────────────────
        feature_df = pd.read_parquet(out_path)

        # Restore timestamp as a plain column if the pipeline made it the index
        if "timestamp" not in feature_df.columns:
            if feature_df.index.name == "timestamp":
                feature_df = feature_df.reset_index()
            else:
                feature_df = feature_df.reset_index(drop=False)

        logger.info(
            "build_inference_features: %d rows x %d cols  [%s → %s]  symbol=%s",
            feature_df.shape[0], feature_df.shape[1],
            feature_df["timestamp"].iloc[0] if "timestamp" in feature_df.columns else "?",
            feature_df["timestamp"].iloc[-1] if "timestamp" in feature_df.columns else "?",
            symbol,
        )
        return feature_df

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"m15_df is missing required columns: {missing}. "
            f"Available: {sorted(df.columns.tolist())}"
        )
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError(
            f"m15_df['timestamp'] must be datetime64 (UTC). "
            f"Got dtype={df['timestamp'].dtype}. "
            "Convert with: df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)"
        )
