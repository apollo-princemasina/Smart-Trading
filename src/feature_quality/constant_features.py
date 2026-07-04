"""Detect constant and near-constant features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConstantReport:
    constant_features:     list[str]   # std == 0 (or single unique value)
    near_constant_features: list[str]  # unique ratio < unique_ratio_threshold
    unique_ratios:         pd.Series   # n_unique / n_rows per feature


class ConstantFeatureDetector:
    """
    Flag features that carry no information.

    Parameters
    ----------
    unique_ratio_threshold:
        Features whose (n_unique_values / n_rows) ratio is below this are
        near-constant (default 0.01 = less than 1 % unique values).
    std_threshold:
        Features whose standard deviation is below this are treated as
        constant (default 1e-10).
    """

    def __init__(
        self,
        unique_ratio_threshold: float = 0.01,
        std_threshold:          float = 1e-10,
    ):
        self._unique_ratio_thresh = unique_ratio_threshold
        self._std_thresh          = std_threshold

    def fit(self, df: pd.DataFrame) -> ConstantReport:
        n = len(df)
        if n == 0:
            return ConstantReport([], [], pd.Series(dtype=float))

        constant:     list[str] = []
        near_constant: list[str] = []
        unique_ratios: dict[str, float] = {}

        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                constant.append(col)
                unique_ratios[col] = 0.0
                continue

            n_unique = series.nunique()
            ratio    = n_unique / n
            unique_ratios[col] = ratio

            # Numeric std check
            if pd.api.types.is_numeric_dtype(series):
                std = series.std()
                if std < self._std_thresh or n_unique == 1:
                    constant.append(col)
                    continue

            if n_unique == 1:
                constant.append(col)
            elif ratio < self._unique_ratio_thresh:
                near_constant.append(col)

        return ConstantReport(
            constant_features      = constant,
            near_constant_features = near_constant,
            unique_ratios          = pd.Series(unique_ratios),
        )
