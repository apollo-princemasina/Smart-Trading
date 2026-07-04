"""Recursive Feature Elimination with optional cross-validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold, KFold

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

logger = logging.getLogger(__name__)


@dataclass
class RFEReport:
    selected_features:  list[str]
    feature_ranking:    pd.Series   # 1 = selected, higher = eliminated first
    optimal_n_features: int
    cv_scores:          pd.Series | None   # cross-validated score per n_features
    support:            pd.Series          # bool mask: True = selected


class RFESelector:
    """
    Recursive Feature Elimination (RFE / RFECV).

    Parameters
    ----------
    n_features_to_select:
        Number of features to keep.  If None, uses cross-validation to find
        the optimal count (slower but automatic).
    step:
        Fraction (0 < step < 1) or integer number of features to remove per
        round (default 0.10).
    cv:
        Number of cross-validation folds when *n_features_to_select* is None
        (default 5).
    classification:
        True → classification; False → regression.
    max_samples:
        Cap training rows (default 30 000).
    random_state:
        Seed.
    """

    def __init__(
        self,
        n_features_to_select: int | None = None,
        step:                 float       = 0.10,
        cv:                   int         = 5,
        classification:       bool        = True,
        max_samples:          int         = 30_000,
        random_state:         int         = 42,
    ):
        self._n_select  = n_features_to_select
        self._step      = step
        self._cv        = cv
        self._clf       = classification
        self._max_samp  = max_samples
        self._rng       = random_state

    def fit(self, df: pd.DataFrame, target: pd.Series) -> RFEReport:
        numeric = df.select_dtypes(include=[np.number]).fillna(df.median(numeric_only=True))
        aligned = target.reindex(numeric.index).dropna()
        numeric = numeric.loc[aligned.index]

        rng = np.random.default_rng(self._rng)
        if len(numeric) > self._max_samp:
            idx     = rng.choice(len(numeric), self._max_samp, replace=False)
            numeric = numeric.iloc[idx]
            aligned = aligned.iloc[idx]

        X    = numeric.values.astype(float)
        y    = aligned.values
        cols = list(numeric.columns)

        estimator = self._build_estimator()

        cv_scores = None
        try:
            if self._n_select is None:
                kfold = (
                    StratifiedKFold(n_splits=self._cv, shuffle=True, random_state=self._rng)
                    if self._clf else
                    KFold(n_splits=self._cv, shuffle=True, random_state=self._rng)
                )
                selector  = RFECV(
                    estimator,
                    step=self._step,
                    cv=kfold,
                    scoring="roc_auc" if self._clf else "r2",
                    n_jobs=-1,
                )
                selector.fit(X, y)
                n_opt      = int(selector.n_features_)
                cv_scores  = pd.Series(
                    selector.cv_results_["mean_test_score"],
                    name="cv_score",
                )
            else:
                n_opt     = self._n_select
                selector  = RFE(estimator, n_features_to_select=n_opt, step=self._step)
                selector.fit(X, y)

            support  = pd.Series(selector.support_, index=cols)
            ranking  = pd.Series(selector.ranking_, index=cols)
            selected = list(ranking[ranking == 1].index)

        except Exception as exc:
            logger.warning("RFE failed: %s — returning all features", exc)
            support  = pd.Series(True, index=cols)
            ranking  = pd.Series(1, index=cols)
            selected = cols
            n_opt    = len(cols)

        return RFEReport(
            selected_features  = selected,
            feature_ranking    = ranking,
            optimal_n_features = n_opt,
            cv_scores          = cv_scores,
            support            = support,
        )

    def _build_estimator(self):
        if _HAS_LGBM:
            cls = LGBMClassifier if self._clf else LGBMRegressor
            return cls(
                n_estimators=100, random_state=self._rng, verbose=-1, n_jobs=-1
            )
        cls = RandomForestClassifier if self._clf else RandomForestRegressor
        return cls(n_estimators=50, random_state=self._rng, n_jobs=-1)
