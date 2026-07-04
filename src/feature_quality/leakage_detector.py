"""
Leakage detection for time-series feature datasets.

Detects
-------
* Future-prefix features used as non-label inputs
* High correlation between a feature and a forward-shifted target (look-ahead)
* Near-perfect correlation with a known label/future column
* Improper alignment: feature correlates more strongly with lead(target) than target
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LeakageReport:
    leakage_scores:   pd.Series          # 0 = clean, 1 = definite leakage
    leakage_types:    dict[str, str]     # feature → detected leakage type
    flagged_features: list[str]          # any leakage detected
    details:          dict[str, dict]    # per-feature detailed info


_LEAKAGE_PREFIXES = ("future_",)
_LABEL_PREFIXES   = ("label_", "future_")


class LeakageDetector:
    """
    Automatically detect data leakage in a feature DataFrame.

    Checks
    ------
    1. **Prefix leakage** — columns with ``future_`` prefix are forward-looking
       and should only appear as labels, never as model inputs.
    2. **Target leakage** — features that are near-perfectly correlated with the
       target (|corr| > *correlation_threshold*) likely encode target information.
    3. **Look-ahead bias** — shift analysis: if |corr(feature, target.shift(-k))| >
       |corr(feature, target)| for small k, the feature may contain future data.

    Parameters
    ----------
    correlation_threshold:
        Correlation with the target above which a feature is flagged (default 0.90).
    max_lead_k:
        Maximum number of bars to lead the target for shift analysis (default 5).
    future_prefix:
        Column prefix that indicates a forward-looking feature (default ``"future_"``).
    label_prefix:
        Column prefix for labels — these are excluded from the input feature set.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.90,
        max_lead_k:            int   = 5,
        future_prefix:         str   = "future_",
        label_prefix:          str   = "label_",
    ):
        self._corr_thresh    = correlation_threshold
        self._k              = max_lead_k
        self._future_prefix  = future_prefix
        self._label_prefix   = label_prefix

    def fit(
        self,
        df:     pd.DataFrame,
        target: pd.Series | None = None,
    ) -> LeakageReport:
        """
        Analyse *df* for leakage.

        Parameters
        ----------
        df:
            Feature DataFrame (must NOT include the target as a column).
        target:
            Optional target Series (aligned with *df* index).
        """
        scores:  dict[str, float] = {}
        types:   dict[str, str]   = {}
        details: dict[str, dict]  = {}

        for col in df.columns:
            score, ltype, det = self._analyse_column(df, col, target)
            scores[col]  = score
            types[col]   = ltype
            details[col] = det

        scores_series = pd.Series(scores)
        flagged       = list(scores_series[scores_series > 0.0].index)

        return LeakageReport(
            leakage_scores   = scores_series,
            leakage_types    = {k: v for k, v in types.items() if v},
            flagged_features = flagged,
            details          = details,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _analyse_column(
        self,
        df:     pd.DataFrame,
        col:    str,
        target: pd.Series | None,
    ) -> tuple[float, str, dict]:
        det: dict = {}

        # ── Check 1: future_ prefix ───────────────────────────────────────────
        if col.startswith(self._future_prefix):
            return (1.0, "future_prefix", {"reason": "Column has future_ prefix"})

        # ── Check 2: correlation with known label/future columns ──────────────
        label_cols = [
            c for c in df.columns
            if c != col and any(
                c.startswith(p) for p in (_LABEL_PREFIXES)
            )
        ]
        max_label_corr = 0.0
        correlated_label = ""
        for lc in label_cols:
            try:
                c = df[col].dropna().corr(df[lc].dropna())
                if abs(c) > abs(max_label_corr):
                    max_label_corr   = c
                    correlated_label = lc
            except Exception:
                pass
        det["max_label_corr"] = round(float(max_label_corr), 4)

        if abs(max_label_corr) >= self._corr_thresh:
            return (
                min(abs(max_label_corr), 1.0),
                "label_correlation",
                {**det, "correlated_with": correlated_label},
            )

        # ── Check 3: target shift analysis ───────────────────────────────────
        if target is not None:
            aligned = target.reindex(df.index)
            try:
                base_corr = float(df[col].corr(aligned))
            except Exception:
                base_corr = 0.0

            det["base_target_corr"] = round(float(base_corr), 4)
            max_lead_corr = base_corr

            for k in range(1, self._k + 1):
                try:
                    lead_corr = float(df[col].corr(aligned.shift(-k)))
                except Exception:
                    continue
                if abs(lead_corr) > abs(max_lead_corr):
                    max_lead_corr    = lead_corr
                    det["max_lead_k"] = k
                    det["lead_corr"]  = round(float(lead_corr), 4)

            # Look-ahead: lead correlation stronger than current
            if (
                abs(max_lead_corr) > abs(base_corr) * 1.10    # >10% improvement
                and abs(max_lead_corr) >= 0.50                 # at least moderate
            ):
                score = min(abs(max_lead_corr), 1.0)
                return (score, "look_ahead_bias", det)

            # Direct target leakage
            if abs(base_corr) >= self._corr_thresh:
                return (abs(base_corr), "target_leakage", det)

        return (0.0, "", det)
