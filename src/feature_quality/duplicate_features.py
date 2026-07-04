"""Detect exact and near-exact duplicate feature columns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DuplicateReport:
    duplicate_pairs:  list[tuple[str, str]]   # (keep, drop)
    features_to_drop: list[str]               # the redundant copy
    duplicate_of:     dict[str, str]           # feature → original


class DuplicateFeatureDetector:
    """
    Identify duplicate columns using correlation-based (|corr| == 1) and
    content-hash comparisons.

    Parameters
    ----------
    corr_threshold:
        Absolute Pearson correlation above which two numeric features are
        considered duplicates (default 1.0 = exact copies only).
    check_hash:
        Also compare column hashes for exact non-numeric equality.
    """

    def __init__(self, corr_threshold: float = 1.0, check_hash: bool = True):
        self._thresh      = corr_threshold
        self._check_hash  = check_hash

    def fit(self, df: pd.DataFrame) -> DuplicateReport:
        pairs:   list[tuple[str, str]] = []
        to_drop: set[str]              = set()
        dup_of:  dict[str, str]        = {}

        numeric = df.select_dtypes(include=[np.number]).dropna()

        # -- Correlation check (numeric columns) --------------------------------
        if not numeric.empty:
            corr = numeric.corr().abs()
            cols  = list(corr.columns)
            for i, a in enumerate(cols):
                for j in range(i + 1, len(cols)):
                    b = cols[j]
                    if b in to_drop:
                        continue
                    if corr.at[a, b] >= self._thresh:
                        pairs.append((a, b))
                        to_drop.add(b)
                        dup_of[b] = a

        # -- Hash check (all columns, catches non-numeric / already-numeric) ---
        if self._check_hash:
            hashes: dict[str, str] = {}
            for col in df.columns:
                if col in to_drop:
                    continue
                h = _col_hash(df[col])
                if h in hashes:
                    orig = hashes[h]
                    if col not in dup_of:  # not already caught by corr
                        pairs.append((orig, col))
                        to_drop.add(col)
                        dup_of[col] = orig
                else:
                    hashes[h] = col

        return DuplicateReport(
            duplicate_pairs  = pairs,
            features_to_drop = sorted(to_drop),
            duplicate_of     = dup_of,
        )


def _col_hash(s: pd.Series) -> str:
    """Quick hash of a column's values (ignores index)."""
    try:
        arr  = s.values
        data = arr.tobytes() if hasattr(arr, "tobytes") else str(arr).encode()
        import hashlib
        return hashlib.md5(data).hexdigest()
    except Exception:
        return str(hash(tuple(s.values[:100])))
