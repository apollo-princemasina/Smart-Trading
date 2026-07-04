"""
Trade Management Labels
========================
For every bar treated as a potential trade entry, simulate the forward
PnL curve (in R-multiples) and derive optimal management actions.

PnL in R-multiples
-------------------
A value of +1.0 means the trade has moved 1 × initial-risk in favour.
A value of -1.0 means the trade has hit the stop-loss distance.

Label columns (per entry bar)
------------------------------
mgmt_strategy         : int  0=simple  1=trail  2=scale_out  3=early_exit
mgmt_optimal_exit_bar : int  relative bar at which PnL is maximum
mgmt_max_r_multiple   : float peak PnL in R-multiples
mgmt_breakeven_bar    : int  first bar trade is profitable (R > 0), or -1
mgmt_trail_bar        : int  first bar R ≥ trail_threshold, or -1
mgmt_partial_exit_bar : int  first bar R ≥ partial_threshold, or -1
mgmt_exit_type        : int  0=tp  1=sl  2=early  3=timeout

Management strategy selection
------------------------------
SIMPLE     (0): TP hit before trailing threshold → just hold to TP
TRAIL      (1): TP hit and trade reached trail_threshold → trail SL
SCALE_OUT  (2): TP hit and trade reached partial_threshold → scale out
EARLY_EXIT (3): SL hit or MAE severe → exit early if signal worsens
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

# Strategy codes
SIMPLE     = 0
TRAIL      = 1
SCALE_OUT  = 2
EARLY_EXIT = 3

# Exit type codes
EXIT_TP      = 0
EXIT_SL      = 1
EXIT_EARLY   = 2
EXIT_TIMEOUT = 3


@dataclass
class TradeManagementConfig:
    atr_period:        int   = 14
    tp_atr_mult:       float = 2.0
    sl_atr_mult:       float = 1.0
    max_bars:          int   = 50
    min_atr_pct:       float = 0.0001
    trail_threshold:   float = 0.50   # R-multiple at which trail is activated
    partial_threshold: float = 0.30   # R-multiple for partial position exit
    early_exit_r:      float = -0.70  # R below this → early exit label


@dataclass
class TradeManagementLabels:
    labels:         pd.DataFrame
    config:         TradeManagementConfig
    simple_pct:     float
    trail_pct:      float
    scale_pct:      float
    early_exit_pct: float
    n_rows:         int
    n_valid:        int


class TradeManagementLabeler:
    """Generate trade-management labels for every potential entry bar."""

    def __init__(self, config: Optional[TradeManagementConfig] = None) -> None:
        self.config = config or TradeManagementConfig()

    def fit(self, df: pd.DataFrame) -> TradeManagementLabels:
        self._validate(df)
        df    = df.copy()
        cfg   = self.config
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr   = compute_atr(df, cfg.atr_period).values
        n     = len(df)
        mb    = cfg.max_bars
        fwd   = np.log(np.roll(close, -1) / close)
        fwd[-1] = 0.0

        strategy_arr  = np.full(n, np.nan)
        opt_exit_arr  = np.full(n, np.nan)
        max_r_arr     = np.full(n, np.nan)
        be_bar_arr    = np.full(n, np.nan)
        trail_arr     = np.full(n, np.nan)
        partial_arr   = np.full(n, np.nan)
        exit_type_arr = np.full(n, np.nan)

        for i in range(n - 1):
            a = atr[i]
            if np.isnan(a) or a < close[i] * cfg.min_atr_pct:
                continue
            entry     = close[i]
            direction = 1 if fwd[i] >= 0 else -1
            risk_dist = cfg.sl_atr_mult * a
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

            outcome, dur, _mfe, _mae = simulate_trade(fh, fl, tp, sl, entry, direction)

            # Build PnL curve in R-multiples using bar mid-prices
            mid = (fh + fl) / 2.0
            if direction == 1:
                r_curve = (mid - entry) / max(risk_dist, 1e-10)
            else:
                r_curve = (entry - mid) / max(risk_dist, 1e-10)

            r_curve = r_curve[:dur]  # trim to actual trade duration

            max_r       = float(r_curve.max()) if len(r_curve) else 0.0
            opt_bar     = int(np.argmax(r_curve)) if len(r_curve) else 0

            # Breakeven bar (first bar R > 0)
            be_candidates = np.where(r_curve > 0)[0]
            be_bar = int(be_candidates[0]) if len(be_candidates) else -1

            # Trail bar (first bar R ≥ trail_threshold)
            tr_cands = np.where(r_curve >= cfg.trail_threshold)[0]
            trail_bar = int(tr_cands[0]) if len(tr_cands) else -1

            # Partial exit bar (first bar R ≥ partial_threshold)
            pt_cands = np.where(r_curve >= cfg.partial_threshold)[0]
            partial_bar = int(pt_cands[0]) if len(pt_cands) else -1

            # Management strategy
            if outcome == TP_FIRST:
                if trail_bar >= 0:
                    strategy = TRAIL
                elif partial_bar >= 0:
                    strategy = SCALE_OUT
                else:
                    strategy = SIMPLE
            elif outcome == SL_FIRST:
                strategy = EARLY_EXIT
            else:
                # Timeout — classify by max R achieved
                strategy = TRAIL if max_r >= cfg.trail_threshold else SIMPLE

            # Exit type
            if outcome == TP_FIRST:
                exit_type = EXIT_TP
            elif outcome == SL_FIRST:
                # Was there an early exit signal? (R dropped below early_exit_r)
                early = any(r_curve[:opt_bar] < cfg.early_exit_r) if opt_bar > 0 else False
                exit_type = EXIT_EARLY if early else EXIT_SL
            else:
                exit_type = EXIT_TIMEOUT

            strategy_arr[i]  = strategy
            opt_exit_arr[i]  = opt_bar
            max_r_arr[i]     = max_r
            be_bar_arr[i]    = be_bar
            trail_arr[i]     = trail_bar
            partial_arr[i]   = partial_bar
            exit_type_arr[i] = exit_type

        # Mark last mb rows as NaN
        strategy_arr[-mb:] = np.nan

        idx    = df.index
        result = pd.DataFrame({
            "mgmt_strategy":         strategy_arr,
            "mgmt_optimal_exit_bar": opt_exit_arr,
            "mgmt_max_r_multiple":   max_r_arr,
            "mgmt_breakeven_bar":    be_bar_arr,
            "mgmt_trail_bar":        trail_arr,
            "mgmt_partial_exit_bar": partial_arr,
            "mgmt_exit_type":        exit_type_arr,
        }, index=idx)

        result.iloc[-mb:] = np.nan

        valid = result["mgmt_strategy"].notna()
        n_ok  = int(valid.sum())
        strat = result["mgmt_strategy"].dropna()

        def _pct(code):
            return float((strat == code).mean()) if len(strat) else 0.0

        logger.info(
            "TradeManagement: SIMPLE=%.1f%% TRAIL=%.1f%% SCALE_OUT=%.1f%% EARLY=%.1f%%",
            _pct(SIMPLE)*100, _pct(TRAIL)*100, _pct(SCALE_OUT)*100, _pct(EARLY_EXIT)*100,
        )
        return TradeManagementLabels(
            labels=result, config=cfg,
            simple_pct=_pct(SIMPLE), trail_pct=_pct(TRAIL),
            scale_pct=_pct(SCALE_OUT), early_exit_pct=_pct(EARLY_EXIT),
            n_rows=n, n_valid=n_ok,
        )

    def _validate(self, df: pd.DataFrame) -> None:
        for col in ("high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
