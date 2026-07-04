"""
Boruta feature selection — custom implementation using sklearn RandomForest.

Algorithm (Kursa & Rudnicki 2010)
----------------------------------
1. Create shadow features: randomly shuffle every feature column.
2. Train RandomForest on (original + shadow) features.
3. For each original feature, compare its importance to the *maximum* shadow
   importance using a two-sided binomial test.
4. Accept features significantly better than shadows; reject those significantly
   worse; mark the rest as tentative.
5. Remove decided features and repeat until convergence or max_iter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
try:
    from scipy.stats import binomtest as _binomtest
    def _binom_test(k, n, p, alternative):
        return _binomtest(k, n, p, alternative=alternative).pvalue
except ImportError:
    from scipy.stats import binom_test as _binom_test_legacy
    def _binom_test(k, n, p, alternative):
        return _binom_test_legacy(k, n, p, alternative=alternative)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


@dataclass
class BorutaReport:
    accepted:          list[str]
    rejected:          list[str]
    tentative:         list[str]
    hit_counts:        pd.Series   # number of rounds feature outperformed max shadow
    n_trials:          pd.Series   # total rounds each feature participated in
    acceptance_pvals:  pd.Series   # p-value (accept test, lower = more important)


class BorutaSelector:
    """
    Boruta all-relevant feature selection.

    Parameters
    ----------
    n_estimators:
        Number of trees in each RandomForest (default 100).
    max_iter:
        Maximum number of Boruta rounds (default 100).
    alpha:
        Significance level for the binomial test (default 0.05).
    classification:
        True → classification; False → regression.
    max_samples:
        Subsample training data to at most this many rows (default 50 000).
    random_state:
        Seed.
    """

    def __init__(
        self,
        n_estimators:   int   = 100,
        max_iter:       int   = 100,
        alpha:          float = 0.05,
        classification: bool  = True,
        max_samples:    int   = 50_000,
        random_state:   int   = 42,
    ):
        self._n_est   = n_estimators
        self._iters   = max_iter
        self._alpha   = alpha
        self._clf     = classification
        self._max_smp = max_samples
        self._rng     = random_state

    def fit(self, df: pd.DataFrame, target: pd.Series) -> BorutaReport:
        numeric = df.select_dtypes(include=[np.number]).fillna(df.median(numeric_only=True))
        aligned = target.reindex(numeric.index).dropna()
        numeric = numeric.loc[aligned.index]

        rng = np.random.default_rng(self._rng)
        if len(numeric) > self._max_smp:
            idx     = rng.choice(len(numeric), self._max_smp, replace=False)
            numeric = numeric.iloc[idx]
            aligned = aligned.iloc[idx]

        X    = numeric.values.astype(float)
        y    = aligned.values
        cols = list(numeric.columns)
        n_f  = len(cols)

        # State tracking
        hit_counts = np.zeros(n_f, dtype=int)
        n_trials   = np.zeros(n_f, dtype=int)
        accepted   = np.zeros(n_f, dtype=bool)
        rejected   = np.zeros(n_f, dtype=bool)

        for it in range(self._iters):
            # Features still undecided
            undecided = ~(accepted | rejected)
            if not undecided.any():
                break

            # Create shadow features (shuffle each column independently)
            X_shadow = _create_shadow(X, rng)
            X_aug    = np.hstack([X, X_shadow])

            # Train model
            model = self._build_model(rng)
            model.fit(X_aug, y)

            imp     = model.feature_importances_
            orig_imp   = imp[:n_f]
            shadow_imp = imp[n_f:]
            max_shadow = shadow_imp.max()

            # Count hits
            for i in range(n_f):
                if undecided[i]:
                    n_trials[i] += 1
                    if orig_imp[i] > max_shadow:
                        hit_counts[i] += 1

            # Binomial test after min 20 rounds per feature
            for i in range(n_f):
                if accepted[i] or rejected[i]:
                    continue
                n = n_trials[i]
                if n < 20:
                    continue
                k   = hit_counts[i]
                # Accept: better than chance (p > 0.5)
                p_acc = float(_binom_test(k, n, 0.5, "greater"))
                # Reject: worse than chance
                p_rej = float(_binom_test(k, n, 0.5, "less"))
                if p_acc < self._alpha:
                    accepted[i] = True
                elif p_rej < self._alpha:
                    rejected[i] = True

            if (it + 1) % 20 == 0:
                n_acc = int(accepted.sum())
                n_rej = int(rejected.sum())
                logger.debug(
                    "Boruta iter %d: accepted=%d rejected=%d tentative=%d",
                    it + 1, n_acc, n_rej, n_f - n_acc - n_rej,
                )

        acc_list  = [cols[i] for i in range(n_f) if accepted[i]]
        rej_list  = [cols[i] for i in range(n_f) if rejected[i]]
        tent_list = [cols[i] for i in range(n_f) if not accepted[i] and not rejected[i]]

        # Compute acceptance p-values for all features
        pvals: dict[str, float] = {}
        for i, col in enumerate(cols):
            n = n_trials[i]
            k = hit_counts[i]
            if n == 0:
                pvals[col] = 1.0
            else:
                pvals[col] = float(_binom_test(k, n, 0.5, "greater"))

        return BorutaReport(
            accepted         = acc_list,
            rejected         = rej_list,
            tentative        = tent_list,
            hit_counts       = pd.Series(hit_counts, index=cols),
            n_trials         = pd.Series(n_trials, index=cols),
            acceptance_pvals = pd.Series(pvals),
        )

    def _build_model(self, rng):
        seed = int(rng.integers(0, 2**31))
        if self._clf:
            return RandomForestClassifier(
                n_estimators=self._n_est,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            )
        return RandomForestRegressor(
            n_estimators=self._n_est,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )


def _create_shadow(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a column-wise shuffled copy of X (shadow features)."""
    shadow = X.copy()
    for j in range(shadow.shape[1]):
        rng.shuffle(shadow[:, j])
    return shadow
