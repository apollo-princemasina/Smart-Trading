"""
Setup Quality Labels
====================
For each bar, evaluate the quality of entering a trade at that bar's close.

Quality is determined by simulating the forward trade (ATR-based TP/SL) and
scoring based on three factors:

    - Did TP hit before SL?           (50 points)
    - How clean was the trade?        MFE/MAE ratio  (30 points max)
    - Was full RR achieved?           actual vs target RR  (20 points max)

Grade thresholds (configurable)
--------------------------------
HIGH   (3)  : score ≥ 70
MEDIUM (2)  : score ≥ 50
LOW    (1)  : score ≥ 25
NO TRADE (0): score < 25

Label columns
-------------
setup_quality      : int   0-3 categorical grade
setup_score        : float [0, 100] raw composite score
setup_mfe_mae_ratio: float MFE / max(MAE, ε)
setup_achievable_rr: float realised RR vs target RR ratio
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .trade_outcome import (
    TP_FIRST, SL_FIRST, compute_atr, simulate_trade,
)

logger = logging.getLogger(__name__)

# Quality grade codes
NO_TRADE = 0
LOW      = 1
MEDIUM   = 2
HIGH     = 3


@dataclass
class SetupQualityConfig:
    atr_period:    int   = 14
    tp_atr_mult:   float = 2.0
    sl_atr_mult:   float = 1.0
    max_bars:      int   = 50
    min_atr_pct:   float = 0.0001
    # Grade cutoffs (score 0–100)
    high_threshold:   float = 70.0
    medium_threshold: float = 50.0
    low_threshold:    float = 25.0


@dataclass
class SetupQualityLabels:
    labels:         pd.DataFrame
    config:         SetupQualityConfig
    high_pct:       float
    medium_pct:     float
    low_pct:        float
    no_trade_pct:   float
    n_rows:         int
    n_valid:        int


class SetupQualityLabeler:
    """Score every bar as a potential trade setup."""

    def __init__(self, config: Optional[SetupQualityConfig] = None) -> None:
        self.config = config or SetupQualityConfig()

    def fit(self, df: pd.DataFrame) -> SetupQualityLabels:
        self._validate(df)
        df    = df.copy()
        cfg   = self.config
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr   = compute_atr(df, cfg.atr_period).values
        n     = len(df)
        mb    = cfg.max_bars
        target_rr = cfg.tp_atr_mult / max(cfg.sl_atr_mult, 1e-8)

        quality_arr = np.full(n, np.nan)
        score_arr   = np.full(n, np.nan)
        mmr_arr     = np.full(n, np.nan)   # MFE/MAE ratio
        arr_arr     = np.full(n, np.nan)   # achievable RR ratio

        # Forward return for direction selection
        fwd = np.log(np.roll(close, -1) / close)
        fwd[-1] = 0.0

        for i in range(n - 1):
            a = atr[i]
            if np.isnan(a) or a < close[i] * cfg.min_atr_pct:
                continue
            entry     = close[i]
            direction = 1 if fwd[i] >= 0 else -1
            end       = min(i + mb + 1, n)
            fh        = high[i + 1 : end]
            fl        = low[i + 1 : end]
            if len(fh) == 0:
                continue

            if direction == 1:
                tp = entry + cfg.tp_atr_mult * a
                sl = entry - cfg.sl_atr_mult * a
            else:
                tp = entry - cfg.tp_atr_mult * a
                sl = entry + cfg.sl_atr_mult * a

            outcome, _dur, mfe, mae = simulate_trade(fh, fl, tp, sl, entry, direction)

            mfe_mae_ratio = mfe / max(mae, 1e-8)

            # TP component (0 or 50)
            tp_pts = 50.0 if outcome == TP_FIRST else 0.0

            # MFE/MAE component (0–30)
            mmr_pts = 30.0 * min(mfe_mae_ratio / 3.0, 1.0)

            # Achievable-RR component (0–20)
            risk_dist = abs(entry - sl)
            reward_dist = mfe * entry   # mfe is already a fraction
            actual_rr = reward_dist / max(risk_dist, 1e-10)
            rr_ratio  = actual_rr / max(target_rr, 1e-8)
            rr_pts    = 20.0 * min(rr_ratio, 1.0)

            score = tp_pts + mmr_pts + rr_pts

            quality = (
                HIGH     if score >= cfg.high_threshold   else
                MEDIUM   if score >= cfg.medium_threshold else
                LOW      if score >= cfg.low_threshold    else
                NO_TRADE
            )

            quality_arr[i] = quality
            score_arr[i]   = score
            mmr_arr[i]     = mfe_mae_ratio
            arr_arr[i]     = rr_ratio

        # Mark last mb rows as NaN
        quality_arr[-mb:] = np.nan
        score_arr[-mb:]   = np.nan

        idx    = df.index
        result = pd.DataFrame({
            "setup_quality":       quality_arr,
            "setup_score":         score_arr,
            "setup_mfe_mae_ratio": mmr_arr,
            "setup_achievable_rr": arr_arr,
        }, index=idx)

        valid  = result["setup_quality"].notna()
        n_ok   = int(valid.sum())
        grades = result["setup_quality"].dropna()

        def _pct(grade):
            return float((grades == grade).mean()) if len(grades) else 0.0

        logger.info(
            "SetupQuality: HIGH=%.1f%% MED=%.1f%% LOW=%.1f%% NONE=%.1f%%",
            _pct(HIGH)*100, _pct(MEDIUM)*100, _pct(LOW)*100, _pct(NO_TRADE)*100,
        )
        return SetupQualityLabels(
            labels=result, config=cfg,
            high_pct=_pct(HIGH), medium_pct=_pct(MEDIUM),
            low_pct=_pct(LOW), no_trade_pct=_pct(NO_TRADE),
            n_rows=n, n_valid=n_ok,
        )

    def _validate(self, df: pd.DataFrame) -> None:
        for col in ("high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
