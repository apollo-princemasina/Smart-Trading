"""
Feature stability analysis — evaluate how consistently a feature ranks
across rolling time windows and market regimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

logger = logging.getLogger(__name__)

# ── Regime labels ──────────────────────────────────────────────────────────────
REGIMES = ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]


@dataclass
class StabilityReport:
    importance_cv:        pd.Series           # coefficient of variation per feature
    stability_scores:     pd.Series           # 0–100 (100 = perfectly stable)
    rolling_importance:   pd.DataFrame        # window × feature
    regime_stability:     dict[str, pd.Series]  # regime label → importance series
    unstable_features:    list[str]           # CV > cv_threshold
    n_windows:            int


class StabilityAnalyzer:
    """
    Measure how stable feature importance is across rolling time windows.

    A feature that is consistently important across many windows is stable.
    High coefficient of variation (CV) of importance = unstable.

    Parameters
    ----------
    n_windows:
        Number of rolling windows (default 10).
    window_frac:
        Fraction of the dataset in each window (default 0.30).
    cv_threshold:
        Features with CV above this are flagged as unstable (default 0.80).
    max_samples_per_window:
        Maximum rows per window for model fitting (default 10 000).
    classification:
        True → classification; False → regression.
    random_state:
        Seed.
    """

    def __init__(
        self,
        n_windows:               int   = 10,
        window_frac:             float = 0.30,
        cv_threshold:            float = 0.80,
        max_samples_per_window:  int   = 10_000,
        classification:          bool  = True,
        random_state:            int   = 42,
    ):
        self._n_windows    = n_windows
        self._win_frac     = window_frac
        self._cv_thresh    = cv_threshold
        self._max_smp      = max_samples_per_window
        self._clf          = classification
        self._rng          = random_state

    def fit(self, df: pd.DataFrame, target: pd.Series) -> StabilityReport:
        numeric = df.select_dtypes(include=[np.number]).fillna(df.median(numeric_only=True))
        aligned = target.reindex(numeric.index).dropna()
        numeric = numeric.loc[aligned.index]

        n    = len(numeric)
        cols = list(numeric.columns)

        win_size  = max(100, int(n * self._win_frac))
        step_size = max(1, (n - win_size) // max(1, self._n_windows - 1))
        starts    = list(range(0, n - win_size + 1, step_size))
        starts    = starts[: self._n_windows]

        rolling_imps: list[pd.Series] = []

        for start in starts:
            end = start + win_size
            X_w = numeric.iloc[start:end].values.astype(float)
            y_w = aligned.iloc[start:end].values

            # Subsample if needed
            if len(X_w) > self._max_smp:
                rng = np.random.default_rng(self._rng + start)
                idx = rng.choice(len(X_w), self._max_smp, replace=False)
                X_w = X_w[idx]
                y_w = y_w[idx]

            imp = self._window_importance(X_w, y_w, cols)
            rolling_imps.append(imp)

        rolling_df = pd.DataFrame(rolling_imps, columns=cols)

        # Coefficient of variation: std / (mean + eps)
        mean_imp = rolling_df.mean()
        std_imp  = rolling_df.std()
        cv       = std_imp / (mean_imp.abs() + 1e-10)

        stability_scores  = (1.0 - cv.clip(0, 1)) * 100.0
        unstable          = list(cv[cv > self._cv_thresh].index)

        # Regime analysis (if DatetimeIndex available)
        regime_stability = self._regime_analysis(numeric, aligned, cols)

        return StabilityReport(
            importance_cv      = cv,
            stability_scores   = stability_scores,
            rolling_importance = rolling_df,
            regime_stability   = regime_stability,
            unstable_features  = unstable,
            n_windows          = len(rolling_imps),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _window_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cols: list[str],
    ) -> pd.Series:
        try:
            model = self._build_model()
            model.fit(X, y)
            return pd.Series(model.feature_importances_, index=cols)
        except Exception as exc:
            logger.debug("Window importance failed: %s", exc)
            return pd.Series(0.0, index=cols)

    def _build_model(self):
        if _HAS_LGBM:
            cls = LGBMClassifier if self._clf else LGBMRegressor
            return cls(n_estimators=50, random_state=self._rng, verbose=-1, n_jobs=-1)
        cls = RandomForestClassifier if self._clf else RandomForestRegressor
        return cls(n_estimators=30, random_state=self._rng, n_jobs=-1)

    def _regime_analysis(
        self,
        df:     pd.DataFrame,
        target: pd.Series,
        cols:   list[str],
    ) -> dict[str, pd.Series]:
        """Compute importance per simple regime label (requires DatetimeIndex)."""
        regime_imp: dict[str, pd.Series] = {}

        if not isinstance(df.index, pd.DatetimeIndex):
            return regime_imp

        try:
            # Simple regime detection using rolling return and volatility
            # Use the first numeric column as a proxy for price
            px = df.iloc[:, 0]

            roll_ret  = px.pct_change(20).fillna(0)
            roll_vol  = px.pct_change().rolling(20).std().fillna(0)

            median_vol = roll_vol.median()
            regimes = pd.Series("ranging", index=df.index)
            regimes[roll_ret > 0.01]  = "trending_up"
            regimes[roll_ret < -0.01] = "trending_down"
            regimes[roll_vol > median_vol * 1.5] = "high_vol"
            regimes[roll_vol < median_vol * 0.5] = "low_vol"

            for label in regimes.unique():
                mask = regimes == label
                X_r  = df[mask].values.astype(float)
                y_r  = target.reindex(df[mask].index).dropna().values

                if len(X_r) < 50 or len(y_r) < 50:
                    continue

                # Align X_r and y_r
                n_min = min(len(X_r), len(y_r))
                X_r, y_r = X_r[:n_min], y_r[:n_min]

                imp = self._window_importance(X_r[:self._max_smp], y_r[:self._max_smp], cols)
                regime_imp[label] = imp

        except Exception as exc:
            logger.debug("Regime analysis failed: %s", exc)

        return regime_imp
