"""
Market Bias Labels
==================
For each bar, generates forward-looking direction and return labels
over configurable horizons (default: 1, 3, 5, 10 bars ahead).

All label columns use future close prices via ``close.shift(-h)``.
The last *h* rows are NaN for each horizon *h* — there is no complete
forward window.  These NaN rows must be dropped before model training.

Label columns produced (for each horizon h)
--------------------------------------------
fwd_return_{h}b   : log-return over h bars (regression)
direction_{h}b    : ternary  0=bearish  1=neutral  2=bullish
bias_{h}b         : binary   0=down  1=up  (neutral → majority class suppressed)
confidence_{h}b   : [0, 1] — |return| / (rolling σ), capped at 1
probability_{h}b  : [0, 1] — sigmoid of z-scored return
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Integer codes for ternary direction
BEARISH = 0
NEUTRAL = 1
BULLISH = 2


@dataclass
class MarketBiasConfig:
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    neutral_threshold: float = 0.0003   # |log-return| < thr → neutral
    price_col: str = "close"
    rolling_vol_window: int = 252       # bars used to normalise confidence


@dataclass
class MarketBiasLabels:
    labels: pd.DataFrame
    config: MarketBiasConfig
    horizon_stats: dict       # {h: {bull_pct, bear_pct, neutral_pct, n_valid}}
    n_rows: int
    n_valid: int              # rows with valid labels for the shortest horizon


class MarketBiasLabeler:
    """Generate market-direction labels at multiple forward horizons."""

    def __init__(self, config: Optional[MarketBiasConfig] = None) -> None:
        self.config = config or MarketBiasConfig()

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> MarketBiasLabels:
        self._validate(df)
        df = df.copy()
        close = df[self.config.price_col].astype(float)
        result = pd.DataFrame(index=df.index)
        stats: dict = {}
        thr = self.config.neutral_threshold

        for h in self.config.horizons:
            # Forward log-return: log(close[i+h] / close[i])
            fwd_ret = np.log(close.shift(-h) / close)
            result[f"fwd_return_{h}b"] = fwd_ret

            # --- Ternary direction ---
            dir_vals = np.where(fwd_ret > thr, BULLISH,
                        np.where(fwd_ret < -thr, BEARISH, NEUTRAL)).astype(float)
            dir_series = pd.Series(dir_vals, index=df.index)
            dir_series[fwd_ret.isna()] = np.nan
            result[f"direction_{h}b"] = dir_series

            # --- Binary bias (0=down, 1=up, ignores magnitude) ---
            binary = (fwd_ret > 0).astype(float)
            binary[fwd_ret.isna()] = np.nan
            result[f"bias_{h}b"] = binary

            # --- Confidence: |return| / rolling-σ, scaled to [0, 1] ---
            roll_std = (
                fwd_ret.rolling(self.config.rolling_vol_window, min_periods=20)
                .std()
                .clip(lower=1e-8)
            )
            confidence = (fwd_ret.abs() / roll_std).clip(0.0, 3.0) / 3.0
            result[f"confidence_{h}b"] = confidence

            # --- Probability: sigmoid of z-score ---
            z = fwd_ret / roll_std
            probability = 1.0 / (1.0 + np.exp(-z.clip(-10, 10)))
            result[f"probability_{h}b"] = probability

            # Per-horizon class distribution
            valid = dir_series.dropna()
            if len(valid):
                stats[h] = {
                    "bull_pct":    float((valid == BULLISH).mean()),
                    "bear_pct":    float((valid == BEARISH).mean()),
                    "neutral_pct": float((valid == NEUTRAL).mean()),
                    "n_valid":     int(valid.notna().sum()),
                }
            else:
                stats[h] = {"bull_pct": 0.0, "bear_pct": 0.0,
                            "neutral_pct": 0.0, "n_valid": 0}

        shortest = self.config.horizons[0]
        n_valid = int(result[f"bias_{shortest}b"].notna().sum())
        logger.info(
            "MarketBias: %d rows, %d valid (shortest horizon=%d)",
            len(df), n_valid, shortest,
        )
        return MarketBiasLabels(
            labels=result,
            config=self.config,
            horizon_stats=stats,
            n_rows=len(df),
            n_valid=n_valid,
        )

    # ------------------------------------------------------------------
    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if self.config.price_col not in df.columns:
            raise ValueError(
                f"price_col '{self.config.price_col}' not found in DataFrame."
            )
        if not self.config.horizons:
            raise ValueError("MarketBiasConfig.horizons must not be empty.")
        for h in self.config.horizons:
            if h < 1:
                raise ValueError(f"All horizons must be >= 1, got {h}.")
