"""
Tree-based feature importance using LightGBM and Random Forest.

LightGBM is used when available (faster, often more accurate).
sklearn RandomForest is always available as a fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

logger = logging.getLogger(__name__)


@dataclass
class ImportanceReport:
    lgbm_importance:     pd.Series | None   # gain importance
    rf_importance:       pd.Series
    combined_importance: pd.Series           # equal-weight average
    top_features:        list[str]
    n_samples_used:      int
    classification:      bool


class TreeImportanceAnalyzer:
    """
    Compute feature importance using LightGBM (if available) and RandomForest.

    Parameters
    ----------
    max_samples:
        Subsample training to at most this many rows (default 50 000).
    n_estimators:
        Number of trees in RandomForest (default 200).
    classification:
        True → binary/multi-class target; False → regression.
    random_state:
        Seed.
    lgbm_n_estimators:
        Number of LightGBM boosting rounds (default 300).
    """

    def __init__(
        self,
        max_samples:       int  = 50_000,
        n_estimators:      int  = 200,
        classification:    bool = True,
        random_state:      int  = 42,
        lgbm_n_estimators: int  = 300,
    ):
        self._max_samp   = max_samples
        self._n_est      = n_estimators
        self._clf        = classification
        self._rng        = random_state
        self._lgbm_est   = lgbm_n_estimators

    def fit(self, df: pd.DataFrame, target: pd.Series) -> ImportanceReport:
        numeric = df.select_dtypes(include=[np.number]).fillna(df.median(numeric_only=True))
        aligned = target.reindex(numeric.index).dropna()
        numeric = numeric.loc[aligned.index]

        rng = np.random.default_rng(self._rng)
        n_used = len(numeric)
        if len(numeric) > self._max_samp:
            idx     = rng.choice(len(numeric), self._max_samp, replace=False)
            numeric = numeric.iloc[idx]
            aligned = aligned.iloc[idx]
            n_used  = self._max_samp

        X    = numeric.values.astype(float)
        y    = aligned.values
        cols = list(numeric.columns)

        # ── LightGBM ──────────────────────────────────────────────────────────
        lgbm_imp = self._fit_lgbm(X, y, cols) if _HAS_LGBM else None

        # ── Random Forest ─────────────────────────────────────────────────────
        rf_imp = self._fit_rf(X, y, cols)

        # ── Combined ──────────────────────────────────────────────────────────
        components = [s for s in [lgbm_imp, rf_imp] if s is not None]
        combined   = _normalise_avg(components, cols)
        top        = list(combined.sort_values(ascending=False).head(50).index)

        return ImportanceReport(
            lgbm_importance     = lgbm_imp,
            rf_importance       = rf_imp,
            combined_importance = combined,
            top_features        = top,
            n_samples_used      = n_used,
            classification      = self._clf,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fit_lgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cols: list[str],
    ) -> pd.Series | None:
        try:
            params = dict(
                n_estimators    = self._lgbm_est,
                random_state    = self._rng,
                n_jobs          = -1,
                importance_type = "gain",
                verbose         = -1,
            )
            if self._clf:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(**params)
            else:
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(**params)
            model.fit(X, y)
            imp = model.feature_importances_
            return pd.Series(imp, index=cols)
        except Exception as exc:
            logger.warning("LightGBM importance failed: %s", exc)
            return None

    def _fit_rf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cols: list[str],
    ) -> pd.Series:
        try:
            if self._clf:
                model = RandomForestClassifier(
                    n_estimators=self._n_est, random_state=self._rng, n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=self._n_est, random_state=self._rng, n_jobs=-1
                )
            model.fit(X, y)
            return pd.Series(model.feature_importances_, index=cols)
        except Exception as exc:
            logger.warning("RandomForest importance failed: %s", exc)
            return pd.Series(0.0, index=cols)


def _normalise_avg(series_list: list[pd.Series], cols: list[str]) -> pd.Series:
    """Normalise each series to [0, 1] and average across all sources."""
    normed = []
    for s in series_list:
        s   = s.reindex(cols).fillna(0.0)
        mx  = s.max()
        normed.append(s / mx if mx > 0 else s)
    avg = pd.concat(normed, axis=1).mean(axis=1)
    return avg.fillna(0.0)
