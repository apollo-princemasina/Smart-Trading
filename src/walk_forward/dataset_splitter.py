"""
Dataset Splitter
================
Slices a DataFrame into train / validation / test subsets using the date
boundaries defined in a ``WindowSpec``.

Guarantees
----------
* Strict chronological ordering is preserved — rows are never reordered.
* No row appears in more than one split.
* Gap bars (if any) are excluded from all splits.
* The original index dtype is preserved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .window_generator import WindowSpec

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """The three splits for a single walk-forward window."""
    window_number: int
    train:         pd.DataFrame
    validation:    pd.DataFrame
    test:          pd.DataFrame

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def val_size(self) -> int:
        return len(self.validation)

    @property
    def test_size(self) -> int:
        return len(self.test)

    def __str__(self) -> str:
        return (
            f"SplitResult(window={self.window_number}, "
            f"train={self.train_size}, val={self.val_size}, test={self.test_size})"
        )


class DatasetSplitter:
    """Split a DataFrame into train/val/test subsets given a WindowSpec."""

    def split(self, df: pd.DataFrame, spec: WindowSpec) -> SplitResult:
        """Slice *df* into three subsets based on the date boundaries in *spec*.

        Args:
            df:   The full dataset with a monotonically increasing DatetimeIndex.
            spec: Window boundary specification.

        Returns:
            SplitResult with three non-overlapping DataFrames.

        Raises:
            TypeError:  If *df* does not have a DatetimeIndex.
            ValueError: If the index is not sorted.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "DatasetSplitter requires a DatetimeIndex. "
                f"Got: {type(df.index).__name__}."
            )
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be monotonically increasing.")

        train = self._slice(df, spec.train_start, spec.train_end)
        val   = self._slice(df, spec.val_start,   spec.val_end)
        test  = self._slice(df, spec.test_start,  spec.test_end)

        logger.debug(
            "Window %03d split: train=%d, val=%d, test=%d rows",
            spec.window_number, len(train), len(val), len(test),
        )

        return SplitResult(
            window_number=spec.window_number,
            train=train,
            validation=val,
            test=test,
        )

    @staticmethod
    def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return rows where start <= index <= end, preserving original order."""
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask].copy()
