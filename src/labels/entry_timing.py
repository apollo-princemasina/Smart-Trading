"""
Entry Timing Labels
====================
For each bar, determine whether this is the optimal entry point within a
forward-looking comparison window.

Algorithm
---------
1. Compute an "entry quality score" for each bar:
       score = tp_probability × smoothness × speed
   where
       tp_probability = 1.0 if TP_FIRST,  0.3 if TIMEOUT,  0.0 if SL_FIRST
       smoothness     = 1 / (1 + mae_pct)         (lower MAE is better)
       speed          = 1 - outcome_bars / max_bars  (faster outcome is better)

2. For every bar i, look forward window_size bars and compare:
       - ENTER_NOW (2): score[i] ≥ enter_threshold × max(scores in window)
       - WAIT      (1): a better entry exists later in the window
       - IGNORE    (0): max(scores in window) < ignore_threshold

Label columns
-------------
entry_signal          : int  0=ignore  1=wait  2=enter_now
optimal_entry_offset  : int  bars ahead to optimal entry (0=this bar, -1=ignore)
time_to_entry         : int  bars until optimal entry (same as offset when >0)
is_optimal_entry      : int  1 if this bar is the best in its window, else 0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .trade_outcome import (
    TP_FIRST, SL_FIRST, TIMEOUT, compute_atr, simulate_trade,
)

logger = logging.getLogger(__name__)

IGNORE     = 0
WAIT       = 1
ENTER_NOW  = 2


@dataclass
class EntryTimingConfig:
    atr_period:       int   = 14
    tp_atr_mult:      float = 2.0
    sl_atr_mult:      float = 1.0
    max_bars:         int   = 50
    window_size:      int   = 10     # look-ahead window for comparison
    enter_threshold:  float = 0.80   # score ≥ 80 % of window-max → enter now
    ignore_threshold: float = 0.10   # window-max below this → ignore
    min_atr_pct:      float = 0.0001


@dataclass
class EntryTimingLabels:
    labels:        pd.DataFrame
    config:        EntryTimingConfig
    enter_pct:     float
    wait_pct:      float
    ignore_pct:    float
    n_rows:        int
    n_valid:       int


class EntryTimingLabeler:
    """Label each bar as Enter-Now, Wait, or Ignore for a timing model."""

    def __init__(self, config: Optional[EntryTimingConfig] = None) -> None:
        self.config = config or EntryTimingConfig()

    def fit(self, df: pd.DataFrame) -> EntryTimingLabels:
        self._validate(df)
        df    = df.copy()
        cfg   = self.config
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr   = compute_atr(df, cfg.atr_period).values
        n     = len(df)
        mb    = cfg.max_bars
        ws    = cfg.window_size
        fwd   = np.log(np.roll(close, -1) / close)
        fwd[-1] = 0.0

        # Step 1: compute raw entry quality score for every bar
        raw_score = np.full(n, np.nan)

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

            outcome, dur, mfe, mae = simulate_trade(fh, fl, tp, sl, entry, direction)

            tp_prob   = 1.0 if outcome == TP_FIRST else (0.3 if outcome == TIMEOUT else 0.0)
            smoothness = 1.0 / (1.0 + mae)
            speed      = 1.0 - dur / max(mb, 1)
            raw_score[i] = tp_prob * smoothness * speed

        # Mark last max(mb, ws) rows as NaN
        lookback = max(mb, ws)
        raw_score[-lookback:] = np.nan

        # Step 2: assign signal based on window comparison
        signal_arr  = np.full(n, np.nan)
        offset_arr  = np.full(n, np.nan)
        optimal_arr = np.zeros(n)

        for i in range(n - lookback):
            if np.isnan(raw_score[i]):
                continue
            win_end   = min(i + ws, n)
            window    = raw_score[i : win_end]
            valid_win = window[~np.isnan(window)]
            if len(valid_win) == 0:
                signal_arr[i] = IGNORE
                offset_arr[i] = -1
                continue

            win_max  = float(valid_win.max())
            best_rel = int(np.nanargmax(window))  # relative offset

            if win_max < cfg.ignore_threshold:
                signal_arr[i] = IGNORE
                offset_arr[i] = -1
            elif raw_score[i] >= cfg.enter_threshold * win_max:
                signal_arr[i]  = ENTER_NOW
                offset_arr[i]  = 0
                optimal_arr[i] = 1
            else:
                signal_arr[i] = WAIT
                offset_arr[i] = best_rel
                if best_rel > 0:
                    optimal_arr[min(i + best_rel, n - 1)] = 1

        idx    = df.index
        result = pd.DataFrame({
            "entry_signal":         signal_arr,
            "optimal_entry_offset": offset_arr,
            "time_to_entry":        np.where(offset_arr >= 0, offset_arr, np.nan),
            "is_optimal_entry":     optimal_arr.astype(float),
        }, index=idx)

        # NaN the tail
        result.iloc[-lookback:] = np.nan

        valid = result["entry_signal"].notna()
        n_ok  = int(valid.sum())
        sig   = result["entry_signal"].dropna()

        def _pct(code):
            return float((sig == code).mean()) if len(sig) else 0.0

        logger.info(
            "EntryTiming: ENTER=%.1f%% WAIT=%.1f%% IGNORE=%.1f%%",
            _pct(ENTER_NOW)*100, _pct(WAIT)*100, _pct(IGNORE)*100,
        )
        return EntryTimingLabels(
            labels=result, config=cfg,
            enter_pct=_pct(ENTER_NOW), wait_pct=_pct(WAIT), ignore_pct=_pct(IGNORE),
            n_rows=n, n_valid=n_ok,
        )

    def _validate(self, df: pd.DataFrame) -> None:
        for col in ("high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
