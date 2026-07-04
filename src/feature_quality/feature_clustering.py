"""Cluster similar features using hierarchical agglomerative clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class ClusterReport:
    cluster_labels:          pd.Series           # feature → cluster_id
    cluster_representatives: dict[int, str]       # cluster_id → representative feature
    n_clusters:              int
    cluster_sizes:           dict[int, int]
    clusters:                dict[int, list[str]]  # cluster_id → feature names


class FeatureClusterer:
    """
    Group highly-correlated features into clusters using hierarchical
    clustering on a correlation-distance matrix.

    Within each cluster a single *representative* feature is selected
    (highest mean absolute correlation with other cluster members).

    Parameters
    ----------
    correlation_threshold:
        Features with |Pearson correlation| ≥ this are placed in the same
        cluster (default 0.70).
    linkage_method:
        Linkage method for ``scipy.cluster.hierarchy.linkage``
        (default ``"average"``).
    max_features:
        Limit analysis to at most this many features (default 300).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.70,
        linkage_method:        str   = "average",
        max_features:          int   = 300,
    ):
        self._thresh      = correlation_threshold
        self._linkage_mth = linkage_method
        self._max_feats   = max_features

    def fit(self, df: pd.DataFrame) -> ClusterReport:
        numeric = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
        numeric = numeric.fillna(numeric.median())

        if numeric.shape[1] > self._max_feats:
            numeric = numeric.iloc[:, : self._max_feats]

        cols = list(numeric.columns)

        if len(cols) < 2:
            single_cluster = {0: cols}
            return ClusterReport(
                cluster_labels          = pd.Series({c: 0 for c in cols}),
                cluster_representatives = {0: cols[0]} if cols else {},
                n_clusters              = 1 if cols else 0,
                cluster_sizes           = {0: len(cols)},
                clusters                = single_cluster,
            )

        corr = numeric.corr(method="pearson").abs()
        # Distance = 1 - |corr|; clamp to [0, 1]
        # Fill NaN (constant columns have undefined correlation) with 1.0 (max distance)
        dist_matrix = (1.0 - corr.values.clip(0, 1)).clip(0)
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=0.0)
        np.fill_diagonal(dist_matrix, 0.0)

        # Condensed distance matrix
        condensed = squareform(dist_matrix, checks=False)
        Z         = linkage(condensed, method=self._linkage_mth)

        # Cut tree at distance = 1 - threshold
        cut_dist    = 1.0 - self._thresh
        labels_arr  = fcluster(Z, t=cut_dist, criterion="distance")
        labels      = pd.Series(labels_arr, index=cols)

        cluster_ids   = sorted(labels.unique())
        clusters:      dict[int, list[str]]  = {}
        reps:          dict[int, str]        = {}
        sizes:         dict[int, int]        = {}

        for cid in cluster_ids:
            members = list(labels[labels == cid].index)
            clusters[cid] = members
            sizes[cid]    = len(members)
            reps[cid]     = self._representative(corr, members)

        return ClusterReport(
            cluster_labels          = labels,
            cluster_representatives = reps,
            n_clusters              = len(cluster_ids),
            cluster_sizes           = sizes,
            clusters                = clusters,
        )

    @staticmethod
    def _representative(corr: pd.DataFrame, members: list[str]) -> str:
        """Return the member with the highest mean absolute correlation to others."""
        if len(members) == 1:
            return members[0]
        sub   = corr.loc[members, members]
        means = sub.sum(axis=1) - 1.0   # subtract self-correlation
        return str(means.idxmax())
