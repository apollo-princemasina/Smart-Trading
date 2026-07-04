"""SHAP-based feature explainability."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


@dataclass
class SHAPReport:
    mean_abs_shap:    pd.Series             # mean |SHAP| per feature
    shap_values:      np.ndarray | None     # raw SHAP matrix (samples × features)
    feature_names:    list[str]
    top_positive:     list[str]             # features with highest avg SHAP
    top_negative:     list[str]             # features with most negative avg SHAP
    interaction_pairs: list[tuple[str, str, float]]  # top feature interactions
    available:        bool = True            # False if shap not installed


class SHAPAnalyzer:
    """
    Compute SHAP values for feature explainability.

    Uses a LightGBM or RandomForest model trained internally.
    Falls back gracefully when the *shap* package is not installed.

    Parameters
    ----------
    max_samples:
        Maximum number of rows used for model training and SHAP computation
        (default 10 000).
    classification:
        True → classification; False → regression.
    random_state:
        Seed.
    """

    def __init__(
        self,
        max_samples:    int  = 10_000,
        classification: bool = True,
        random_state:   int  = 42,
    ):
        self._max_samp = max_samples
        self._clf      = classification
        self._rng      = random_state

    def fit(
        self,
        df:     pd.DataFrame,
        target: pd.Series,
        model   = None,
    ) -> SHAPReport:
        if not _HAS_SHAP:
            logger.warning("shap package not installed — skipping SHAP analysis")
            cols = list(df.select_dtypes(include=[np.number]).columns)
            return SHAPReport(
                mean_abs_shap  = pd.Series(0.0, index=cols),
                shap_values    = None,
                feature_names  = cols,
                top_positive   = [],
                top_negative   = [],
                interaction_pairs = [],
                available      = False,
            )

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

        # ── Train model ───────────────────────────────────────────────────────
        if model is None:
            model = self._build_model()

        split = int(len(X) * 0.80)
        model.fit(X[:split], y[:split])
        X_explain = X[split:] if len(X) > 10 else X

        # ── SHAP values ───────────────────────────────────────────────────────
        try:
            explainer  = _shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_explain)

            # For binary classification, shap_values returns list[arr] → take class 1
            if isinstance(shap_vals, list):
                shap_matrix = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
            else:
                shap_matrix = shap_vals

            mean_abs   = pd.Series(
                np.abs(shap_matrix).mean(axis=0), index=cols
            )
            mean_signed = pd.Series(
                shap_matrix.mean(axis=0), index=cols
            )
            top_pos    = list(mean_signed.sort_values(ascending=False).head(20).index)
            top_neg    = list(mean_signed.sort_values(ascending=True).head(20).index)

            # Top interaction pairs (by product of mean abs SHAP)
            interactions = self._top_interactions(mean_abs, n=10)

        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            shap_matrix  = None
            mean_abs     = pd.Series(0.0, index=cols)
            top_pos, top_neg, interactions = [], [], []

        return SHAPReport(
            mean_abs_shap     = mean_abs,
            shap_values       = shap_matrix,
            feature_names     = cols,
            top_positive      = top_pos,
            top_negative      = top_neg,
            interaction_pairs = interactions,
            available         = True,
        )

    def save_summary_parquet(
        self,
        report: SHAPReport,
        path:   Path,
    ) -> None:
        """Save SHAP summary as a Parquet file."""
        if report.shap_values is None:
            return
        df = pd.DataFrame(
            report.shap_values,
            columns=report.feature_names,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_model(self):
        if _HAS_LGBM:
            cls = LGBMClassifier if self._clf else LGBMRegressor
            return cls(n_estimators=100, random_state=self._rng, verbose=-1, n_jobs=-1)
        cls = RandomForestClassifier if self._clf else RandomForestRegressor
        return cls(n_estimators=50, random_state=self._rng, n_jobs=-1)

    @staticmethod
    def _top_interactions(
        mean_abs: pd.Series,
        n: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Return top-n feature pairs ranked by product of mean |SHAP|."""
        feats  = list(mean_abs.sort_values(ascending=False).head(30).index)
        pairs: list[tuple[str, str, float]] = []
        for i, a in enumerate(feats):
            for b in feats[i + 1:]:
                score = float(mean_abs[a] * mean_abs[b])
                pairs.append((a, b, score))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]
