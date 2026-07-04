"""Permutation-based feature importance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, r2_score


@dataclass
class PermImportanceReport:
    mean_importance: pd.Series
    std_importance:  pd.Series
    top_features:    list[str]
    baseline_score:  float


class PermutationImportanceAnalyzer:
    """
    Compute model-agnostic permutation importance.

    A feature's importance is measured as the drop in model performance
    when that feature's values are randomly shuffled.

    Parameters
    ----------
    n_repeats:
        Number of shuffles per feature (default 10).
    max_samples:
        Maximum number of training rows (default 20 000).
    n_estimators:
        Number of trees in the internal Random Forest (default 100).
    classification:
        True → classification; False → regression.
    random_state:
        Seed.
    """

    def __init__(
        self,
        n_repeats:      int  = 10,
        max_samples:    int  = 20_000,
        n_estimators:   int  = 100,
        classification: bool = True,
        random_state:   int  = 42,
    ):
        self._n_repeats  = n_repeats
        self._max_samp   = max_samples
        self._n_est      = n_estimators
        self._clf        = classification
        self._rng        = random_state

    def fit(
        self,
        df:     pd.DataFrame,
        target: pd.Series,
        model   = None,
    ) -> PermImportanceReport:
        numeric = df.select_dtypes(include=[np.number]).fillna(df.median(numeric_only=True))
        aligned = target.reindex(numeric.index).dropna()
        numeric = numeric.loc[aligned.index]

        # Subsample
        rng = np.random.default_rng(self._rng)
        if len(numeric) > self._max_samp:
            idx     = rng.choice(len(numeric), self._max_samp, replace=False)
            numeric = numeric.iloc[idx]
            aligned = aligned.iloc[idx]

        X = numeric.values.astype(float)
        y = aligned.values

        split   = int(len(X) * 0.80)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if model is None:
            if self._clf:
                model = RandomForestClassifier(
                    n_estimators=self._n_est, random_state=self._rng, n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=self._n_est, random_state=self._rng, n_jobs=-1
                )

        model.fit(X_train, y_train)

        # Baseline score
        try:
            if self._clf:
                prob = model.predict_proba(X_test)[:, 1]
                baseline = float(roc_auc_score(y_test, prob))
            else:
                baseline = float(r2_score(y_test, model.predict(X_test)))
        except Exception:
            baseline = 0.0

        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=self._n_repeats,
            random_state=self._rng,
            n_jobs=-1,
        )

        mean_imp = pd.Series(result.importances_mean, index=numeric.columns)
        std_imp  = pd.Series(result.importances_std,  index=numeric.columns)
        top      = list(mean_imp[mean_imp > 0].sort_values(ascending=False).index)

        return PermImportanceReport(
            mean_importance = mean_imp,
            std_importance  = std_imp,
            top_features    = top,
            baseline_score  = baseline,
        )
