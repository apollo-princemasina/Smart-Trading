"""Convert model predictions into actionable trading signals.

Typical usage
-------------
    from src.inference.signal_generator import generate_signals

    signals = generate_signals(predictions, probabilities, feature_df)
    for sig in signals:
        print(sig)  # Signal(timestamp=..., direction='BUY', confidence=0.82, ...)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Only emit a signal when the model's max-class probability exceeds this threshold.
# Set to 0.0 to emit a signal for every bar (including low-confidence HOLD).
DEFAULT_MIN_CONFIDENCE = 0.60

# Only emit BUY / SELL signals (never HOLD).
DEFAULT_DIRECTIONAL_ONLY = True

CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass
class Signal:
    """A single actionable trading signal."""
    timestamp:   object           # UTC-aware datetime
    direction:   str              # "BUY", "SELL", or "HOLD"
    pred_class:  int              # 0, 1, or 2
    confidence:  float            # max class probability
    prob_sell:   float
    prob_hold:   float
    prob_buy:    float
    close:       Optional[float] = None   # last close price at signal time
    atr:         Optional[float] = None   # ATR at signal time (for TP/SL sizing)
    extra:       dict = field(default_factory=dict)

    def __str__(self) -> str:
        atr_str = f"  atr={self.atr:.5f}" if self.atr is not None else ""
        close_str = f"  close={self.close:.5f}" if self.close is not None else ""
        return (
            f"Signal({self.direction:4s}  conf={self.confidence:.2%}"
            f"  [S={self.prob_sell:.2f} H={self.prob_hold:.2f} B={self.prob_buy:.2f}]"
            f"{close_str}{atr_str}  @ {self.timestamp})"
        )


def generate_signals(
    predictions:    np.ndarray,
    probabilities:  np.ndarray,
    feature_df:     pd.DataFrame,
    *,
    min_confidence:    float = DEFAULT_MIN_CONFIDENCE,
    directional_only:  bool  = DEFAULT_DIRECTIONAL_ONLY,
) -> list[Signal]:
    """Convert raw model outputs into Signal objects.

    Parameters
    ----------
    predictions : np.ndarray, shape (n_rows,)
        Class labels from ``predictor.predict()``.  0=SELL, 1=HOLD, 2=BUY.
    probabilities : np.ndarray, shape (n_rows, 3)
        Class probabilities from ``predictor.predict()``.
    feature_df : pd.DataFrame
        The feature DataFrame used to produce *predictions*.
        Used to extract timestamps, close prices, and ATR values.
    min_confidence : float
        Discard signals whose max-class probability is below this threshold.
        Default 0.60 (60 %).
    directional_only : bool
        If True (default), HOLD predictions are suppressed — only BUY and SELL
        signals are emitted.

    Returns
    -------
    list[Signal]
        Filtered list of Signal objects, one per qualifying bar.
        Empty list if no bars pass the confidence and direction filters.
    """
    n = len(predictions)
    if n == 0:
        return []

    if probabilities.shape != (n, 3):
        raise ValueError(
            f"probabilities shape {probabilities.shape} does not match "
            f"predictions length {n}.  Expected ({n}, 3)."
        )

    # Pull optional scalar columns from the feature DataFrame
    ts_vals    = _col(feature_df, "timestamp", n)
    close_vals = _col(feature_df, "close",     n)
    atr_vals   = _col(feature_df, "atr",       n)

    signals: list[Signal] = []
    for i in range(n):
        pred_class  = int(predictions[i])
        direction   = CLASS_NAMES[pred_class]
        prob_sell   = float(probabilities[i, 0])
        prob_hold   = float(probabilities[i, 1])
        prob_buy    = float(probabilities[i, 2])
        confidence  = float(probabilities[i].max())

        if confidence < min_confidence:
            continue
        if directional_only and direction == "HOLD":
            continue

        signals.append(Signal(
            timestamp  = ts_vals[i],
            direction  = direction,
            pred_class = pred_class,
            confidence = confidence,
            prob_sell  = prob_sell,
            prob_hold  = prob_hold,
            prob_buy   = prob_buy,
            close      = close_vals[i],
            atr        = atr_vals[i],
        ))

    logger.info(
        "generate_signals: %d qualifying signals from %d bars "
        "(min_conf=%.0f%%, directional_only=%s)",
        len(signals), n, min_confidence * 100, directional_only,
    )
    return signals


def latest_signal(
    predictions:   np.ndarray,
    probabilities: np.ndarray,
    feature_df:    pd.DataFrame,
    *,
    min_confidence:   float = DEFAULT_MIN_CONFIDENCE,
    directional_only: bool  = DEFAULT_DIRECTIONAL_ONLY,
) -> Optional[Signal]:
    """Return only the most recent qualifying signal (last bar), or None.

    Convenience wrapper used by the live inference loop: call once per M15
    bar close to get the single actionable signal for the new bar.
    """
    all_signals = generate_signals(
        predictions, probabilities, feature_df,
        min_confidence=min_confidence,
        directional_only=directional_only,
    )
    return all_signals[-1] if all_signals else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str, n: int) -> list:
    """Extract a column as a plain list, or return [None]*n if absent."""
    if name in df.columns:
        vals = df[name].tolist()
        return [v if not _is_nan(v) else None for v in vals]
    return [None] * n


def _is_nan(v: object) -> bool:
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False
