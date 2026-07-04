"""Variance Inflation Factor and multicollinearity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VIFReport:
    vif_scores:        pd.Series    # VIF per feature (lower = better)
    tolerance:         pd.Series    # 1 / VIF
    condition_number:  float        # overall condition number of the feature matrix
    high_vif_features: list[str]    # VIF > threshold
    recommendations:   list[str]    # human-readable removal suggestions


def _compute_vif_single(X: np.ndarray, idx: int) -> float:
    """
    VIF for column *idx* in matrix *X*.

    VIF_j = 1 / (1 − R²_j) where R²_j is from regressing column j on all others.
    """
    y     = X[:, idx]
    mask  = list(range(X.shape[1]))
    mask.pop(idx)
    X_oth = X[:, mask]

    # Add intercept
    ones   = np.ones((X_oth.shape[0], 1))
    X_oth  = np.hstack([ones, X_oth])

    try:
        coeffs, *_ = np.linalg.lstsq(X_oth, y, rcond=None)
        y_hat   = X_oth @ coeffs
        ss_res  = np.sum((y - y_hat) ** 2)
        ss_tot  = np.sum((y - y.mean()) ** 2)
        r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r2      = max(0.0, min(r2, 1.0 - 1e-10))   # clamp to avoid div/0
        return float(1.0 / (1.0 - r2))
    except Exception:
        return float("inf")


class VIFAnalyzer:
    """
    Compute Variance Inflation Factor (VIF) for multicollinearity detection.

    VIF interpretation
    ------------------
    * VIF < 5   : acceptable
    * 5 ≤ VIF < 10 : moderate multicollinearity (consider removing)
    * VIF ≥ 10  : severe multicollinearity (remove)

    Parameters
    ----------
    threshold:
        Features with VIF above this are flagged (default 10.0).
    max_features:
        Limit the number of columns analysed (VIF is O(n_cols²) slow).
        Default 100.
    """

    def __init__(self, threshold: float = 10.0, max_features: int = 100):
        self._thresh    = threshold
        self._max_feats = max_features

    def fit(self, df: pd.DataFrame) -> VIFReport:
        numeric = df.select_dtypes(include=[np.number]).dropna()
        numeric = numeric.fillna(numeric.median())

        if numeric.shape[1] > self._max_feats:
            numeric = numeric.iloc[:, : self._max_feats]

        # Remove zero-variance columns before computing VIF
        std = numeric.std()
        numeric = numeric.loc[:, std > 1e-10]

        if numeric.shape[1] < 2:
            empty = pd.Series(dtype=float)
            return VIFReport(empty, empty, 0.0, [], [])

        X    = numeric.values.astype(float)
        cols = list(numeric.columns)

        vif_vals = [_compute_vif_single(X, i) for i in range(len(cols))]
        vif      = pd.Series(vif_vals, index=cols)
        tol      = (1.0 / vif.replace(0, np.inf)).fillna(0.0)
        flagged  = list(vif[vif > self._thresh].index)

        # Condition number of (X^T X)
        try:
            eig = np.linalg.eigvalsh(X.T @ X)
            eig = np.abs(eig)
            cond = float(np.sqrt(eig.max() / (eig.min() + 1e-12)))
        except Exception:
            cond = float("nan")

        recommendations = [
            f"Consider removing '{f}' (VIF={vif[f]:.1f} > {self._thresh})"
            for f in flagged
        ]

        return VIFReport(
            vif_scores        = vif,
            tolerance         = tol,
            condition_number  = cond,
            high_vif_features = flagged,
            recommendations   = recommendations,
        )
