"""Detect missing values, infinite values, and invalid entries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MissingValueReport:
    missing_counts:   pd.Series
    missing_rates:    pd.Series
    infinite_counts:  pd.Series
    infinite_rates:   pd.Series
    total_invalid:    pd.Series   # NaN + inf per feature
    flagged_missing:  list[str]   # > missing_threshold
    flagged_infinite: list[str]   # any infinite


class MissingValueAnalyzer:
    """
    Detect NaN and infinite values in every column of a DataFrame.

    Parameters
    ----------
    missing_threshold:
        Features with NaN rate > this are flagged (default 0.30).
    """

    def __init__(self, missing_threshold: float = 0.30):
        self._thresh = missing_threshold

    def fit(self, df: pd.DataFrame) -> MissingValueReport:
        n = len(df)
        numeric = df.select_dtypes(include=[np.number])

        missing_counts  = df.isnull().sum()
        missing_rates   = missing_counts / n

        inf_mask        = numeric.apply(lambda s: np.isinf(s).sum())
        inf_counts      = pd.Series(0, index=df.columns)
        inf_counts.update(inf_mask)
        inf_rates       = inf_counts / n

        total_invalid   = missing_counts + inf_counts

        flagged_missing  = list(missing_rates[missing_rates > self._thresh].index)
        flagged_infinite = list(inf_counts[inf_counts > 0].index)

        return MissingValueReport(
            missing_counts   = missing_counts,
            missing_rates    = missing_rates,
            infinite_counts  = inf_counts,
            infinite_rates   = inf_rates,
            total_invalid    = total_invalid,
            flagged_missing  = flagged_missing,
            flagged_infinite = flagged_infinite,
        )

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def quality_scores(self, report: MissingValueReport) -> pd.Series:
        """
        Return a per-feature quality score 0–100 based on completeness.
        Features with any infinite values are capped at 50.
        """
        scores = (1.0 - report.missing_rates) * 100.0
        scores = scores.clip(0, 100)
        # Penalise infinite values
        inf_penalty = (report.infinite_rates > 0).astype(float) * 50.0
        scores      = (scores - inf_penalty).clip(0, 100)
        return scores
