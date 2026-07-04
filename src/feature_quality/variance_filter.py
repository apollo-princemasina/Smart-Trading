"""Filter features by variance threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VarianceReport:
    variances:        pd.Series
    std_devs:         pd.Series
    below_threshold:  list[str]   # variance < threshold


class VarianceFilter:
    """
    Compute per-feature variance and flag those below a threshold.

    Parameters
    ----------
    threshold:
        Features with variance strictly below this are flagged (default 1e-5).
    """

    def __init__(self, threshold: float = 1e-5):
        self._thresh = threshold

    def fit(self, df: pd.DataFrame) -> VarianceReport:
        numeric = df.select_dtypes(include=[np.number])
        variances = numeric.var(ddof=0)   # population variance
        std_devs  = numeric.std(ddof=0)

        below = list(variances[variances < self._thresh].index)

        return VarianceReport(
            variances       = variances,
            std_devs        = std_devs,
            below_threshold = below,
        )

    def quality_scores(self, report: VarianceReport) -> pd.Series:
        """
        Map variance to a 0–100 quality score using log scaling.
        Variance = 0 → score 0; variance ≥ 1.0 → score 100.
        """
        scores = (np.log1p(report.variances) / np.log1p(1.0)).clip(0, 1) * 100
        scores[report.below_threshold] = 0.0
        return scores
