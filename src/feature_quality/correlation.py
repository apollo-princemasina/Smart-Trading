"""Pearson, Spearman, Kendall, and Distance correlation analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CorrelationReport:
    pearson:           pd.DataFrame
    spearman:          pd.DataFrame
    high_corr_pairs:   list[dict]    # {feat_a, feat_b, method, correlation}
    features_to_drop:  list[str]     # redundant features (one from each pair)
    cluster_groups:    list[list[str]]  # groups of highly-correlated features


class CorrelationAnalyzer:
    """
    Compute pairwise feature correlations and identify redundant features.

    Parameters
    ----------
    threshold:
        Absolute correlation above which a feature pair is flagged as redundant
        (default 0.95).
    max_features:
        Subsample to at most this many features for Spearman (expensive).
        Default 200.
    """

    def __init__(
        self,
        threshold:    float = 0.95,
        max_features: int   = 200,
    ):
        self._thresh      = threshold
        self._max_feats   = max_features

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> CorrelationReport:
        numeric = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
        numeric = numeric.fillna(numeric.median())

        # Subsample columns if too many
        if numeric.shape[1] > self._max_feats:
            numeric = numeric.iloc[:, : self._max_feats]

        pearson  = numeric.corr(method="pearson")
        spearman = numeric.corr(method="spearman")

        high_pairs:  list[dict]  = []
        to_drop:     set[str]    = set()

        cols = list(pearson.columns)
        for i, a in enumerate(cols):
            for j in range(i + 1, len(cols)):
                b     = cols[j]
                p_val = abs(float(pearson.at[a, b]))
                s_val = abs(float(spearman.at[a, b]))
                val   = max(p_val, s_val)
                if val >= self._thresh:
                    high_pairs.append({
                        "feat_a":      a,
                        "feat_b":      b,
                        "pearson":     round(float(pearson.at[a, b]), 4),
                        "spearman":    round(float(spearman.at[a, b]), 4),
                        "max_abs":     round(val, 4),
                    })
                    if b not in to_drop:
                        to_drop.add(b)

        clusters = self._build_clusters(pearson)

        return CorrelationReport(
            pearson          = pearson,
            spearman         = spearman,
            high_corr_pairs  = high_pairs,
            features_to_drop = sorted(to_drop),
            cluster_groups   = clusters,
        )

    # ── Distance correlation (scipy-free, O(n²)) ─────────────────────────────

    @staticmethod
    def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Energy-based distance correlation (Székely & Rizzo 2007).
        Returns a value in [0, 1] (0 = independent, 1 = perfectly dependent).
        """
        n  = len(x)
        if n < 4:
            return float("nan")
        x, y = np.asarray(x, float), np.asarray(y, float)
        a    = np.abs(x[:, None] - x[None, :])
        b    = np.abs(y[:, None] - y[None, :])
        a    = a - a.mean(axis=0) - a.mean(axis=1, keepdims=True) + a.mean()
        b    = b - b.mean(axis=0) - b.mean(axis=1, keepdims=True) + b.mean()
        dcov2_xy = (a * b).mean()
        dcov2_xx = (a * a).mean()
        dcov2_yy = (b * b).mean()
        denom    = np.sqrt(dcov2_xx * dcov2_yy)
        if denom < 1e-12:
            return 0.0
        dcor = float(np.sqrt(max(dcov2_xy / denom, 0.0)))
        return dcor

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_clusters(
        self,
        corr_matrix: pd.DataFrame,
    ) -> list[list[str]]:
        """Group features into clusters of highly-correlated columns."""
        cols = list(corr_matrix.columns)
        seen: set[str]         = set()
        clusters: list[list[str]] = []

        for col in cols:
            if col in seen:
                continue
            group = [col]
            seen.add(col)
            for other in cols:
                if other == col or other in seen:
                    continue
                if abs(corr_matrix.at[col, other]) >= self._thresh:
                    group.append(other)
                    seen.add(other)
            if len(group) > 1:
                clusters.append(group)

        return clusters

    def compute_kendall(self, df: pd.DataFrame, max_cols: int = 50) -> pd.DataFrame:
        """Compute Kendall tau correlation (slow — capped at max_cols features)."""
        numeric = df.select_dtypes(include=[np.number]).fillna(df.median())
        if numeric.shape[1] > max_cols:
            numeric = numeric.iloc[:, :max_cols]
        return numeric.corr(method="kendall")
