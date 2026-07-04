"""
Analytics
=========
Post-backtest analysis that goes beyond trade-level metrics:

  - Market regime detection  (trending / ranging / high_vol / low_vol)
  - Session analysis         (London / New York / Asian / overlap / off-hours)
  - Prediction accuracy by confidence band
  - Directional accuracy over time (rolling window)
  - Per-direction (BUY / SELL) performance breakdown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Session detection ─────────────────────────────────────────────────────────

_SESSIONS: dict[str, tuple[int, int]] = {
    "asian":    (0,  8),    # 00:00–08:00 UTC
    "london":   (7,  16),   # 07:00–16:00 UTC
    "newyork":  (13, 22),   # 13:00–22:00 UTC
}


def get_session(ts: pd.Timestamp) -> str:
    """Return the trading session for a UTC timestamp."""
    h = ts.hour
    in_london  = _SESSIONS["london"][0]  <= h < _SESSIONS["london"][1]
    in_newyork = _SESSIONS["newyork"][0] <= h < _SESSIONS["newyork"][1]
    in_asian   = _SESSIONS["asian"][0]   <= h < _SESSIONS["asian"][1]

    if in_london and in_newyork:
        return "overlap"
    if in_london:
        return "london"
    if in_newyork:
        return "newyork"
    if in_asian:
        return "asian"
    return "offhours"


# ── Market regime detection ───────────────────────────────────────────────────

def classify_regime(
    price_df:   pd.DataFrame,
    bar_idx:    int,
    lookback:   int = 20,
    atr_col:    str = "atr",
    close_col:  str = "close",
) -> str:
    """Classify market regime at bar_idx as 'trending' | 'ranging' | 'high_vol' | 'low_vol'.

    Method: ADX proxy via range / ATR ratio.
    """
    start = max(0, bar_idx - lookback)
    window = price_df.iloc[start: bar_idx + 1]

    if close_col not in window.columns or len(window) < 5:
        return "unknown"

    closes = window[close_col].values.astype(float)
    net_move = abs(closes[-1] - closes[0])
    if atr_col in window.columns:
        atr_mean = float(window[atr_col].mean())
    else:
        highs  = window["high"].values  if "high"  in window.columns else closes
        lows   = window["low"].values   if "low"   in window.columns else closes
        atr_mean = float(np.mean(highs - lows))

    if atr_mean == 0:
        return "unknown"

    # Efficiency ratio: net displacement / sum of bar ranges
    bar_ranges = np.abs(np.diff(closes))
    total_range = float(bar_ranges.sum()) if len(bar_ranges) > 0 else 1.0
    er = net_move / total_range if total_range > 0 else 0.0

    # ATR relative to recent mean
    if atr_col in window.columns and len(window) >= lookback:
        hist_start = max(0, bar_idx - lookback * 2)
        hist = price_df[atr_col].iloc[hist_start: bar_idx + 1]
        hist_mean = float(hist.mean()) if len(hist) > 0 else atr_mean
        vol_ratio = atr_mean / hist_mean if hist_mean > 0 else 1.0
    else:
        vol_ratio = 1.0

    if vol_ratio > 1.5:
        return "high_vol"
    if vol_ratio < 0.6:
        return "low_vol"
    if er > 0.4:
        return "trending"
    return "ranging"


# ── Confidence band analysis ──────────────────────────────────────────────────

def analyze_confidence_bands(
    trades: list,
    bands:  list[float] = [0.60, 0.70, 0.80, 0.90, 1.01],
) -> list[dict]:
    """Return performance stats for each confidence band."""
    results = []
    lower = 0.0
    for upper in bands:
        band_trades = [
            t for t in trades
            if getattr(t, "confidence", 0.0) >= lower
            and getattr(t, "confidence", 0.0) < upper
        ]
        if not band_trades:
            lower = upper
            continue
        profits = [t.net_profit for t in band_trades if t.net_profit is not None]
        winners = [p for p in profits if p > 0]
        results.append({
            "band":      f"[{lower:.2f}, {upper:.2f})",
            "n_trades":  len(band_trades),
            "win_rate":  round(len(winners) / len(profits), 4) if profits else 0.0,
            "net_profit": round(sum(profits), 4) if profits else 0.0,
            "avg_profit": round(sum(profits) / len(profits), 4) if profits else 0.0,
        })
        lower = upper
    return results


# ── Session performance breakdown ────────────────────────────────────────────

def analyze_session_performance(trades: list) -> dict[str, dict]:
    """Group closed trade performance by trading session."""
    groups: dict[str, list[float]] = {}
    for t in trades:
        sess = getattr(t, "session", None) or "unknown"
        if t.net_profit is None:
            continue
        groups.setdefault(sess, []).append(t.net_profit)

    result = {}
    for sess, profits in groups.items():
        wins = [p for p in profits if p > 0]
        result[sess] = {
            "n_trades":   len(profits),
            "win_rate":   round(len(wins) / len(profits), 4) if profits else 0.0,
            "net_profit": round(sum(profits), 4),
            "avg_profit": round(sum(profits) / len(profits), 4) if profits else 0.0,
        }
    return result


# ── Direction performance ─────────────────────────────────────────────────────

def analyze_direction_performance(trades: list) -> dict[str, dict]:
    """Return performance stats separately for BUY and SELL trades."""
    groups: dict[str, list[float]] = {"BUY": [], "SELL": []}
    for t in trades:
        d = getattr(t, "direction", None)
        if d in groups and t.net_profit is not None:
            groups[d].append(t.net_profit)

    result = {}
    for direction, profits in groups.items():
        if not profits:
            continue
        wins = [p for p in profits if p > 0]
        result[direction] = {
            "n_trades":   len(profits),
            "win_rate":   round(len(wins) / len(profits), 4),
            "net_profit": round(sum(profits), 4),
            "avg_profit": round(sum(profits) / len(profits), 4),
        }
    return result


# ── Rolling accuracy ─────────────────────────────────────────────────────────

def compute_rolling_accuracy(
    trades:   list,
    window:   int = 20,
) -> pd.DataFrame:
    """Rolling win-rate over the last `window` closed trades."""
    closed = [t for t in trades if getattr(t, "status", None) == "closed"]
    if not closed:
        return pd.DataFrame()

    rows = [
        {
            "exit_time":  t.exit_time,
            "is_winner":  1 if t.net_profit and t.net_profit > 0 else 0,
        }
        for t in closed
    ]
    df = pd.DataFrame(rows).sort_values("exit_time").set_index("exit_time")
    df["rolling_win_rate"] = df["is_winner"].rolling(window=window, min_periods=1).mean()
    return df[["rolling_win_rate"]]


# ── Regime performance ────────────────────────────────────────────────────────

def analyze_regime_performance(trades: list, price_df: pd.DataFrame) -> dict[str, dict]:
    """Group performance by the market regime at each trade's entry bar."""
    groups: dict[str, list[float]] = {}
    for t in trades:
        bar_idx = getattr(t, "entry_bar_idx", None)
        if bar_idx is None or t.net_profit is None:
            continue
        regime = classify_regime(price_df, bar_idx)
        groups.setdefault(regime, []).append(t.net_profit)

    result = {}
    for regime, profits in groups.items():
        wins = [p for p in profits if p > 0]
        result[regime] = {
            "n_trades":   len(profits),
            "win_rate":   round(len(wins) / len(profits), 4) if profits else 0.0,
            "net_profit": round(sum(profits), 4),
        }
    return result
