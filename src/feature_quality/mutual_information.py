"""Mutual information feature scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)


@dataclass
class MIReport:
    mi_scores:         pd.Series   # raw mutual information per feature
    mi_scores_norm:    pd.Series   # normalised 0–1
    top_features:      list[str]   # features with MI > median


class MutualInformationAnalyzer:
    """
    Compute mutual information between every feature and the target.

    Parameters
    ----------
    classification:
        True → use ``mutual_info_classif`` (binary/multi-class target).
        False → use ``mutual_info_regression`` (continuous target).
    random_state:
        Seed for reproducibility.
    max_samples:
        Subsample the data to at most this many rows for speed (default 50 000).
    n_neighbors:
        k-NN neighbours used by the MI estimator (default 3).
    """

    def __init__(
        self,
        classification: bool = True,
        random_state:   int  = 42,
        max_samples:    int  = 50_000,
        n_neighbors:    int  = 3,
    ):
        self._clf         = classification
        self._rng         = random_state
        self._max_samples = max_samples
        self._k           = n_neighbors

    def fit(self, df: pd.DataFrame, target: pd.Series) -> MIReport:
        numeric = df.select_dtypes(include=[np.number]).copy()
        # Impute NaN with column median
        numeric = numeric.fillna(numeric.median())

        # Align with target
        aligned = target.reindex(numeric.index).dropna()
        numeric = numeric.loc[aligned.index]

        # Subsample
        if len(numeric) > self._max_samples:
            idx     = np.random.default_rng(self._rng).choice(
                len(numeric), self._max_samples, replace=False
            )
            numeric = numeric.iloc[idx]
            aligned = aligned.iloc[idx]

        X = numeric.values.astype(float)
        y = aligned.values

        fn = mutual_info_classif if self._clf else mutual_info_regression
        mi = fn(X, y, random_state=self._rng, n_neighbors=self._k)

        mi_series = pd.Series(mi, index=numeric.columns)
        mi_max    = mi_series.max()
        mi_norm   = mi_series / mi_max if mi_max > 0 else mi_series.copy()
        top       = list(mi_series[mi_series > mi_series.median()].sort_values(ascending=False).index)

        return MIReport(
            mi_scores      = mi_series,
            mi_scores_norm = mi_norm,
            top_features   = top,
        )
