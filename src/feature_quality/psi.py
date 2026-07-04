"""Population Stability Index (PSI) for feature drift detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# PSI interpretation thresholds
PSI_GREEN  = 0.10   # no significant change
PSI_YELLOW = 0.20   # minor shift
# PSI > 0.20 → major shift (red)


@dataclass
class PSIReport:
    psi_scores:       pd.Series           # per-feature PSI
    psi_labels:       pd.Series           # "stable" / "minor_shift" / "major_shift"
    flagged_features: list[str]           # PSI > threshold
    reference_stats:  dict[str, dict]     # bin edges + reference counts
    current_stats:    dict[str, dict]     # bin edges + current counts


def compute_psi(
    reference: np.ndarray,
    current:   np.ndarray,
    bins:      int = 10,
) -> float:
    """
    Compute PSI between *reference* and *current* distributions.

    PSI = Σ (A_i% − E_i%) × ln(A_i% / E_i%)
    where E = reference (expected), A = current (actual).

    Returns
    -------
    float
        PSI score (0 = identical; > 0.2 = significant shift).
    """
    reference = np.asarray(reference, dtype=float)
    current   = np.asarray(current,   dtype=float)
    reference = reference[np.isfinite(reference)]
    current   = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Build bin edges from reference distribution
    edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current,   bins=edges)[0]

    # Convert to proportions
    eps   = 1e-8
    ref_p = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
    cur_p = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))

    psi = np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))
    return float(np.clip(psi, 0.0, None))


def _psi_label(score: float) -> str:
    if score < PSI_GREEN:
        return "stable"
    if score < PSI_YELLOW:
        return "minor_shift"
    return "major_shift"


class PSICalculator:
    """
    Compute PSI for every numeric feature in a DataFrame by splitting it
    into a reference (training) period and a current (test) period.

    Parameters
    ----------
    bins:
        Number of quantile bins (default 10).
    threshold:
        PSI score above which a feature is flagged (default 0.20).
    split_ratio:
        Fraction of rows used as the reference distribution (default 0.70).
    """

    def __init__(
        self,
        bins:        int   = 10,
        threshold:   float = 0.20,
        split_ratio: float = 0.70,
    ):
        self._bins   = bins
        self._thresh = threshold
        self._split  = split_ratio

    def fit(self, df: pd.DataFrame) -> PSIReport:
        numeric = df.select_dtypes(include=[np.number])
        n_ref   = max(1, int(len(numeric) * self._split))

        psi_scores: dict[str, float] = {}
        ref_stats:  dict[str, dict]  = {}
        cur_stats:  dict[str, dict]  = {}

        for col in numeric.columns:
            ref_vals = numeric[col].iloc[:n_ref].values
            cur_vals = numeric[col].iloc[n_ref:].values
            score    = compute_psi(ref_vals, cur_vals, self._bins)
            psi_scores[col] = score
            ref_stats[col]  = {"n": int(np.isfinite(ref_vals).sum())}
            cur_stats[col]  = {"n": int(np.isfinite(cur_vals).sum())}

        psi_series = pd.Series(psi_scores)
        labels     = psi_series.map(_psi_label)
        flagged    = list(psi_series[psi_series > self._thresh].index)

        return PSIReport(
            psi_scores       = psi_series,
            psi_labels       = labels,
            flagged_features = flagged,
            reference_stats  = ref_stats,
            current_stats    = cur_stats,
        )

    def fit_two(
        self,
        df:        pd.DataFrame,
        reference: pd.DataFrame,
    ) -> PSIReport:
        """
        Compute PSI comparing *reference* DataFrame to *df* (current).

        Both DataFrames must share column names.
        """
        common  = list(set(reference.columns) & set(df.columns))
        numeric = df[common].select_dtypes(include=[np.number])
        psi_scores: dict[str, float] = {}

        for col in numeric.columns:
            ref_vals = reference[col].dropna().values
            cur_vals = df[col].dropna().values
            psi_scores[col] = compute_psi(ref_vals, cur_vals, self._bins)

        psi_series = pd.Series(psi_scores)
        labels     = psi_series.map(_psi_label)
        flagged    = list(psi_series[psi_series > self._thresh].index)

        return PSIReport(
            psi_scores       = psi_series,
            psi_labels       = labels,
            flagged_features = flagged,
            reference_stats  = {},
            current_stats    = {},
        )
