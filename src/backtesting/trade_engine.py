"""
Trade Engine
============
Converts ML prediction arrays into TradeSignal objects.

Direction mapping (default)
---------------------------
  Binary (2 classes):   0 → SELL,  1 → BUY
  Ternary (3 classes):  0 → SELL,  1 → HOLD, 2 → BUY
  N-class:              0 → SELL,  N-1 → BUY, rest → HOLD

A signal is downgraded to HOLD when confidence < min_probability.
The engine NEVER modifies predictions — confidence filtering is the only
transformation applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TradeSignal:
    """One trading signal derived from a single bar's ML prediction."""
    bar_idx:          int
    timestamp:        pd.Timestamp
    direction:        str                       # "BUY", "SELL", or "HOLD"
    prediction_class: int                       # raw integer from model
    confidence:       float                     # max class probability
    probabilities:    Optional[list[float]] = field(default=None, repr=False)

    @property
    def is_actionable(self) -> bool:
        return self.direction in ("BUY", "SELL")


class TradeEngine:
    """Convert prediction arrays → TradeSignal list (read-only).

    Args:
        min_probability:      Minimum confidence to act (default 0.60).
        prediction_class_map: Override the default direction mapping.
    """

    def __init__(
        self,
        min_probability:       float = 0.60,
        prediction_class_map:  Optional[dict[int, str]] = None,
    ) -> None:
        self.min_probability      = min_probability
        self._class_map           = prediction_class_map

    def generate_signals(
        self,
        timestamps:    pd.Series,
        predictions:   np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> list[TradeSignal]:
        """Generate one TradeSignal per bar.

        Args:
            timestamps:    Series of bar timestamps (length N).
            predictions:   Integer class predictions (length N).
            probabilities: Optional (N, n_classes) probability matrix.

        Returns:
            List of TradeSignal, one per bar.
        """
        signals: list[TradeSignal] = []
        n = len(predictions)

        for i in range(n):
            pred_class = int(predictions[i])

            if probabilities is not None:
                proba_row  = probabilities[i].tolist()
                confidence = float(np.max(probabilities[i]))
            else:
                proba_row  = None
                confidence = 0.5

            direction = self._map_class(pred_class, probabilities)

            if direction != "HOLD" and confidence < self.min_probability:
                direction = "HOLD"

            signals.append(TradeSignal(
                bar_idx          = i,
                timestamp        = timestamps.iloc[i],
                direction        = direction,
                prediction_class = pred_class,
                confidence       = confidence,
                probabilities    = proba_row,
            ))
        return signals

    def _map_class(
        self,
        pred_class:    int,
        probabilities: Optional[np.ndarray],
    ) -> str:
        if self._class_map is not None:
            return self._class_map.get(pred_class, "HOLD")

        n_classes = probabilities.shape[1] if probabilities is not None else 2

        if n_classes == 2:
            return "BUY" if pred_class == 1 else "SELL"
        if n_classes == 3:
            return {0: "SELL", 1: "HOLD", 2: "BUY"}.get(pred_class, "HOLD")
        # N-class: extremes are BUY/SELL, middle classes are HOLD
        if pred_class == n_classes - 1:
            return "BUY"
        if pred_class == 0:
            return "SELL"
        return "HOLD"
