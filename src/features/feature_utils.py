"""Shared utilities for the feature engineering pipeline.

All helpers are pure functions with no global state.  Import freely from
any module in the ``src.features`` package without risk of circular imports.
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Timing ──────────────────────────────────────────────────────────────────


@contextmanager
def timer(label: str = "") -> Generator[list[float], None, None]:
    """Context manager that records elapsed wall-clock milliseconds.

    Usage::

        with timer("my_feature") as t:
            result = my_heavy_computation()
        elapsed_ms = t[0]   # first element set on exit

    Yields a single-element list so the elapsed time can be read after the
    block without extra variables.
    """
    elapsed: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed[0] = (time.perf_counter() - start) * 1000.0
        if label:
            logger.debug("%s: %.1f ms", label, elapsed[0])


# ── DataFrame fingerprinting ─────────────────────────────────────────────────


def data_fingerprint(df: pd.DataFrame, n_sample: int = 5) -> str:
    """Produce a short hash that identifies a DataFrame's content.

    Takes the first and last *n_sample* rows, serialises them to CSV, and
    hashes the result together with the row count.  Suitable as a cache-key
    component — not cryptographically secure.

    Parameters
    ----------
    df:
        Source DataFrame.
    n_sample:
        Number of rows to take from the head and tail for hashing.

    Returns
    -------
    str
        12-character hexadecimal fingerprint.
    """
    sample   = pd.concat([df.head(n_sample), df.tail(n_sample)], ignore_index=True)
    key_str  = f"rows={len(df)}:{sample.to_csv(index=False)}"
    digest   = hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()
    return digest[:12]


# ── Column management ────────────────────────────────────────────────────────


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return a copy of *df* with all column names prefixed by *prefix_*.

    Example::

        prefix_columns(df, "atr") → columns renamed "atr_value", "atr_pct", …

    Parameters
    ----------
    df:
        DataFrame to rename.
    prefix:
        String prepended to every column name with an underscore separator.

    Returns
    -------
    pd.DataFrame
        New DataFrame with renamed columns and the same index.
    """
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns})


def drop_input_columns(
    output_df: pd.DataFrame,
    input_df:  pd.DataFrame,
) -> pd.DataFrame:
    """Remove any columns from *output_df* that already exist in *input_df*.

    Prevents accidental duplication of OHLCV columns in the feature output.
    """
    overlap = [c for c in output_df.columns if c in input_df.columns]
    if overlap:
        logger.warning(
            "Dropping %d column(s) from feature output that shadow input: %s",
            len(overlap), overlap,
        )
        output_df = output_df.drop(columns=overlap)
    return output_df


# ── DataFrame alignment ──────────────────────────────────────────────────────


def align_to_base(
    feature_df: pd.DataFrame,
    base_df:    pd.DataFrame,
) -> pd.DataFrame:
    """Reindex *feature_df* to exactly match *base_df*'s index.

    Missing rows are filled with NaN.  Extra rows (not in *base_df*) are
    silently dropped.  This corrects for any off-by-one errors in feature
    generators that use rolling / shift operations.
    """
    return feature_df.reindex(base_df.index)


def merge_features(
    base_df:     pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """Horizontally concatenate *base_df* with all non-empty feature DataFrames.

    Parameters
    ----------
    base_df:
        The OHLCV input DataFrame (defines the index).
    feature_dfs:
        List of single-feature output DataFrames returned by each generator.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame.  Duplicate column names are an error — they
        indicate two generators produced the same column name.
    """
    non_empty = [df for df in feature_dfs if not df.empty and df.shape[1] > 0]
    all_frames = [base_df] + non_empty

    combined = pd.concat(all_frames, axis=1)

    dups = combined.columns[combined.columns.duplicated()].tolist()
    if dups:
        raise ValueError(
            f"Duplicate column names detected after merging features: {dups}. "
            "Ensure each feature generator uses a unique column prefix."
        )

    return combined


# ── Required-column validation ───────────────────────────────────────────────


def check_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ``ValueError`` if any *required* column is absent from *df*.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    required:
        List of column names that must be present.

    Raises
    ------
    ValueError
        Listing every missing column.
    """
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


# ── Parquet I/O ──────────────────────────────────────────────────────────────


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to Parquet using PyArrow engine.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination ``.parquet`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    mb = path.stat().st_size / 1_048_576
    logger.info(
        "Saved %d rows x %d cols (%.2f MB) -> %s",
        len(df), df.shape[1], mb, path.name,
    )


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file using PyArrow engine.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {path}\n"
            "Run the preprocessing pipeline first: "
            "python scripts/preprocess_data.py"
        )
    df = pd.read_parquet(path, engine="pyarrow")
    logger.debug("Loaded %d rows x %d cols from %s", len(df), df.shape[1], path.name)
    return df


# ── Feature caching ──────────────────────────────────────────────────────────


def cache_path(
    cache_dir:    Path,
    symbol:       str,
    feature_name: str,
    fingerprint:  str,
) -> Path:
    """Construct a deterministic cache file path.

    Parameters
    ----------
    cache_dir:
        Root cache directory (from ``config.settings.FEATURE_CACHE_DIR``).
    symbol:
        Instrument symbol, e.g. "EURUSD".
    feature_name:
        ``BaseFeature.name`` of the generator.
    fingerprint:
        12-char hash from ``data_fingerprint()``.

    Returns
    -------
    Path
        e.g. ``cache_dir/EURUSD/my_feature_abc123def456.parquet``
    """
    safe_name = feature_name.replace("/", "_").replace("\\", "_")
    return cache_dir / symbol / f"{safe_name}_{fingerprint}.parquet"


def load_from_cache(path: Path) -> pd.DataFrame | None:
    """Return the cached DataFrame if the cache file exists, else ``None``.

    Parameters
    ----------
    path:
        Full path returned by ``cache_path()``.
    """
    if path.exists():
        logger.debug("Cache hit: %s", path.name)
        return pd.read_parquet(path, engine="pyarrow")
    return None


def save_to_cache(df: pd.DataFrame, path: Path) -> None:
    """Write a feature output to the cache directory.

    Parameters
    ----------
    df:
        Feature output DataFrame.
    path:
        Full path returned by ``cache_path()``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.debug("Cached %d rows -> %s", len(df), path.name)


# ── Statistical helpers ──────────────────────────────────────────────────────


def has_infinite_values(df: pd.DataFrame) -> bool:
    """Return ``True`` if any numeric column in *df* contains ``±Inf``."""
    numeric = df.select_dtypes(include=[np.number])
    return bool(np.isinf(numeric.values).any())


def constant_columns(df: pd.DataFrame, tol: float = 1e-10) -> list[str]:
    """Return names of numeric columns whose standard deviation is below *tol*.

    These columns carry no information and will not contribute to ML models.
    """
    numeric = df.select_dtypes(include=[np.number])
    return [col for col in numeric.columns if numeric[col].std() < tol]
