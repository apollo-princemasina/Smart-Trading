"""
Trade Outcome Labels
====================
For every bar (treated as a potential trade entry at the bar's close),
simulate forward price action and compute:

  - Whether Take-Profit or Stop-Loss is hit first
  - Maximum Favourable / Adverse Excursion (MFE / MAE) over the trade
  - Trade duration (bars to outcome)
  - Realised Risk-Reward

TP/SL are ATR-based, making them instrument-agnostic.

Label columns (long)
---------------------
long_outcome        : int   0=timeout  1=tp_first  2=sl_first
long_outcome_bars   : int   bars until outcome (max_bars if timeout)
long_mfe_pct        : float maximum favourable excursion as % of entry
long_mae_pct        : float maximum adverse excursion as % of entry
long_rr             : float target risk-reward ratio (tp_mult / sl_mult)

Identical columns exist for short_ prefix.

Composite columns (direction chosen by 1-bar forward return)
-------------------------------------------------------------
outcome, outcome_bars, mfe_pct, mae_pct, realized_rr
expected_reward_pct, expected_risk_pct, trade_duration_bars
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Outcome codes (stored as int8 in output; exposed as float for NaN support)
TIMEOUT  = 0
TP_FIRST = 1
SL_FIRST = 2

_REQUIRED_COLS = frozenset({"open", "high", "low", "close"})


@dataclass
class TradeOutcomeConfig:
    atr_period:     int   = 14
    tp_atr_mult:    float = 2.0     # TP = entry ± tp_atr × ATR
    sl_atr_mult:    float = 1.0     # SL = entry ∓ sl_atr × ATR
    max_bars:       int   = 50      # maximum trade duration (bars)
    min_atr_pct:    float = 0.0001  # floor ATR as fraction of price


@dataclass
class TradeOutcomeLabels:
    labels:         pd.DataFrame
    config:         TradeOutcomeConfig
    long_tp_rate:   float
    long_sl_rate:   float
    short_tp_rate:  float
    short_sl_rate:  float
    n_rows:         int
    n_valid:        int


# ── Public helpers (importable by sibling modules) ────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder Average True Range."""
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev  = close.shift(1)
    tr    = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, min_periods=max(1, period // 2)).mean()


def simulate_trade(
    f_high:    np.ndarray,
    f_low:     np.ndarray,
    tp:        float,
    sl:        float,
    entry:     float,
    direction: int,        # +1 = long,  -1 = short
) -> tuple[int, int, float, float]:
    """
    Bar-by-bar trade simulation.

    Returns
    -------
    (outcome, duration_bars, mfe_pct, mae_pct)

    When both TP and SL are hit on the same bar, SL takes priority
    (conservative / worst-case assumption).
    """
    n = len(f_high)
    if n == 0:
        return TIMEOUT, 0, 0.0, 0.0

    if direction == 1:       # long
        tp_hit = f_high >= tp
        sl_hit = f_low  <= sl
    else:                    # short
        tp_hit = f_low  <= tp
        sl_hit = f_high >= sl

    tp_idx = int(np.argmax(tp_hit)) if tp_hit.any() else n
    sl_idx = int(np.argmax(sl_hit)) if sl_hit.any() else n

    if not tp_hit.any() and not sl_hit.any():
        outcome  = TIMEOUT
        duration = n
    elif sl_idx <= tp_idx:
        outcome  = SL_FIRST
        duration = sl_idx + 1
    else:
        outcome  = TP_FIRST
        duration = tp_idx + 1

    # MFE / MAE over ACTUAL trade duration only
    dh = f_high[:duration]
    dl = f_low[:duration]

    if direction == 1:
        mfe = float(dh.max() - entry) / entry if len(dh) else 0.0
        mae = float(entry - dl.min()) / entry if len(dl) else 0.0
    else:
        mfe = float(entry - dl.min()) / entry if len(dl) else 0.0
        mae = float(dh.max() - entry) / entry if len(dh) else 0.0

    return outcome, duration, max(0.0, mfe), max(0.0, mae)


# ── Labeler ───────────────────────────────────────────────────────────────────

class TradeOutcomeLabeler:
    """Simulate a long and a short trade from every bar's close price."""

    def __init__(self, config: Optional[TradeOutcomeConfig] = None) -> None:
        self.config = config or TradeOutcomeConfig()

    def fit(self, df: pd.DataFrame) -> TradeOutcomeLabels:
        self._validate(df)
        df   = df.copy()
        cfg  = self.config
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr   = compute_atr(df, cfg.atr_period).values
        n     = len(df)
        mb    = cfg.max_bars

        # Pre-allocate output arrays
        l_outcome  = np.full(n, np.nan)
        l_ob       = np.full(n, np.nan)
        l_mfe      = np.full(n, np.nan)
        l_mae      = np.full(n, np.nan)
        l_rr       = np.full(n, np.nan)

        s_outcome  = np.full(n, np.nan)
        s_ob       = np.full(n, np.nan)
        s_mfe      = np.full(n, np.nan)
        s_mae      = np.full(n, np.nan)
        s_rr       = np.full(n, np.nan)

        floor_atr = cfg.min_atr_pct

        for i in range(n - 1):
            a = atr[i]
            if np.isnan(a) or a < close[i] * floor_atr:
                continue
            entry = close[i]
            end   = min(i + mb + 1, n)
            fh    = high[i + 1 : end]
            fl    = low[i + 1 : end]
            if len(fh) == 0:
                continue

            # Long
            l_tp = entry + cfg.tp_atr_mult * a
            l_sl = entry - cfg.sl_atr_mult * a
            lo, lb, lmfe, lmae = simulate_trade(fh, fl, l_tp, l_sl, entry, 1)
            l_outcome[i] = lo
            l_ob[i]      = lb
            l_mfe[i]     = lmfe
            l_mae[i]     = lmae
            if (entry - l_sl) > 1e-10:
                l_rr[i] = (l_tp - entry) / (entry - l_sl)

            # Short
            s_tp = entry - cfg.tp_atr_mult * a
            s_sl = entry + cfg.sl_atr_mult * a
            so, sb, smfe, smae = simulate_trade(fh, fl, s_tp, s_sl, entry, -1)
            s_outcome[i] = so
            s_ob[i]      = sb
            s_mfe[i]     = smfe
            s_mae[i]     = smae
            if (s_sl - entry) > 1e-10:
                s_rr[i] = (entry - s_tp) / (s_sl - entry)

        # Mark last mb rows as NaN (incomplete forward window)
        l_outcome[-mb:] = np.nan
        s_outcome[-mb:] = np.nan

        idx    = df.index
        result = pd.DataFrame(index=idx)
        result["long_outcome"]      = l_outcome
        result["long_outcome_bars"] = l_ob
        result["long_mfe_pct"]      = l_mfe
        result["long_mae_pct"]      = l_mae
        result["long_rr"]           = l_rr
        result["short_outcome"]     = s_outcome
        result["short_outcome_bars"]= s_ob
        result["short_mfe_pct"]     = s_mfe
        result["short_mae_pct"]     = s_mae
        result["short_rr"]          = s_rr

        # Composite: direction selected by 1-bar forward return
        fwd = np.log(np.roll(close, -1) / close)
        fwd[-1] = 0.0
        is_long = fwd >= 0  # ties → long

        result["outcome"]             = np.where(is_long, l_outcome, s_outcome)
        result["outcome_bars"]        = np.where(is_long, l_ob,      s_ob)
        result["mfe_pct"]             = np.where(is_long, l_mfe,     s_mfe)
        result["mae_pct"]             = np.where(is_long, l_mae,     s_mae)
        result["realized_rr"]         = np.where(is_long, l_rr,      s_rr)
        result["expected_reward_pct"] = (cfg.tp_atr_mult * atr) / close.clip(1e-10)
        result["expected_risk_pct"]   = (cfg.sl_atr_mult * atr) / close.clip(1e-10)
        result["trade_duration_bars"] = result["outcome_bars"]

        # Last mb rows → NaN for composite columns too
        result.iloc[-mb:] = np.nan

        valid  = result["long_outcome"].notna()
        n_ok   = int(valid.sum())
        safe   = n - mb

        long_tp_rate  = float((l_outcome[:safe] == TP_FIRST).sum() / max(safe, 1))
        long_sl_rate  = float((l_outcome[:safe] == SL_FIRST).sum() / max(safe, 1))
        short_tp_rate = float((s_outcome[:safe] == TP_FIRST).sum() / max(safe, 1))
        short_sl_rate = float((s_outcome[:safe] == SL_FIRST).sum() / max(safe, 1))

        logger.info(
            "TradeOutcome: long TP=%.1f%% SL=%.1f%%  short TP=%.1f%% SL=%.1f%%",
            long_tp_rate * 100, long_sl_rate * 100,
            short_tp_rate * 100, short_sl_rate * 100,
        )
        return TradeOutcomeLabels(
            labels=result, config=cfg,
            long_tp_rate=long_tp_rate, long_sl_rate=long_sl_rate,
            short_tp_rate=short_tp_rate, short_sl_rate=short_sl_rate,
            n_rows=n, n_valid=n_ok,
        )

    # ------------------------------------------------------------------
    def _validate(self, df: pd.DataFrame) -> None:
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
