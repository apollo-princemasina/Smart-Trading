"""Feature drift detection: KS test, Jensen-Shannon distance, PSI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from .psi import PSICalculator, compute_psi


@dataclass
class DriftReport:
    ks_statistics:    pd.Series   # KS statistic per feature
    ks_pvalues:       pd.Series   # KS p-value per feature
    js_distances:     pd.Series   # Jensen-Shannon distance per feature
    psi_scores:       pd.Series   # PSI per feature
    drift_labels:     pd.Series   # "stable" / "minor_drift" / "major_drift"
    drifted_features: list[str]   # features flagged by ANY test
    n_reference:      int
    n_current:        int


def _js_distance(ref: np.ndarray, cur: np.ndarray, bins: int = 50) -> float:
    """Jensen-Shannon distance between two 1-D distributions."""
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    all_vals  = np.concatenate([ref, cur])
    edges     = np.linspace(all_vals.min(), all_vals.max(), bins + 1)
    eps       = 1e-10
    p         = np.histogram(ref, bins=edges)[0].astype(float) + eps
    q         = np.histogram(cur, bins=edges)[0].astype(float) + eps
    p        /= p.sum()
    q        /= q.sum()
    return float(jensenshannon(p, q))


class DriftDetector:
    """
    Detect feature drift between a reference period and a test period.

    Parameters
    ----------
    split_ratio:
        Fraction of rows used as reference (training) period (default 0.70).
    ks_alpha:
        p-value threshold for the Kolmogorov-Smirnov test (default 0.05).
    psi_threshold:
        PSI score above which a feature is flagged (default 0.20).
    js_threshold:
        Jensen-Shannon distance above which a feature is flagged (default 0.15).
    """

    def __init__(
        self,
        split_ratio:   float = 0.70,
        ks_alpha:      float = 0.05,
        psi_threshold: float = 0.20,
        js_threshold:  float = 0.15,
    ):
        self._split  = split_ratio
        self._ks_a   = ks_alpha
        self._psi_t  = psi_threshold
        self._js_t   = js_threshold

    def fit(self, df: pd.DataFrame) -> DriftReport:
        numeric = df.select_dtypes(include=[np.number])
        n_ref   = max(1, int(len(numeric) * self._split))
        ref_df  = numeric.iloc[:n_ref]
        cur_df  = numeric.iloc[n_ref:]

        return self.fit_two(ref_df, cur_df)

    def fit_two(
        self,
        reference: pd.DataFrame,
        current:   pd.DataFrame,
    ) -> DriftReport:
        """Compare *reference* period to *current* period explicitly."""
        common  = list(set(reference.columns) & set(current.columns))
        numeric_ref = reference[common].select_dtypes(include=[np.number])
        numeric_cur = current[common].select_dtypes(include=[np.number])

        ks_stats:  dict[str, float] = {}
        ks_pvals:  dict[str, float] = {}
        js_dists:  dict[str, float] = {}
        psi_s:     dict[str, float] = {}

        for col in numeric_ref.columns:
            ref_vals = numeric_ref[col].dropna().values
            cur_vals = numeric_cur[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                ks_stats[col] = float("nan")
                ks_pvals[col] = float("nan")
                js_dists[col] = float("nan")
                psi_s[col]    = float("nan")
                continue

            ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)
            ks_stats[col] = float(ks_stat)
            ks_pvals[col] = float(ks_p)
            js_dists[col] = _js_distance(ref_vals, cur_vals)
            psi_s[col]    = compute_psi(ref_vals, cur_vals)

        ks_series = pd.Series(ks_stats)
        kp_series = pd.Series(ks_pvals)
        js_series = pd.Series(js_dists)
        ps_series = pd.Series(psi_s)

        flagged: set[str] = set()
        flagged.update(kp_series[kp_series < self._ks_a].index)
        flagged.update(ps_series[ps_series > self._psi_t].index)
        flagged.update(js_series[js_series > self._js_t].index)

        labels = ks_series.apply(self._label)

        return DriftReport(
            ks_statistics    = ks_series,
            ks_pvalues       = kp_series,
            js_distances     = js_series,
            psi_scores       = ps_series,
            drift_labels     = labels,
            drifted_features = sorted(flagged - {float("nan")}),
            n_reference      = len(numeric_ref),
            n_current        = len(numeric_cur),
        )

    def _label(self, ks_stat: float) -> str:
        if np.isnan(ks_stat):
            return "unknown"
        if ks_stat < 0.10:
            return "stable"
        if ks_stat < 0.20:
            return "minor_drift"
        return "major_drift"
