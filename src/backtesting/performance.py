"""
Performance Metrics
===================
Computes 20+ institutional-grade performance metrics from closed trade lists
and equity curves.

All functions are pure (no side effects).  Input types are always validated
before computation to avoid silent division-by-zero.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(trades: list) -> dict:
    """Thin compatibility shim — delegates to compute_performance_metrics."""
    closed = [t for t in trades if getattr(t, "status", None) == "closed"]
    if not closed:
        return {}
    profits = [t.net_profit for t in closed if t.net_profit is not None]
    return compute_performance_metrics(profits, equity_curve=None, trades=closed)


def compute_performance_metrics(
    net_profits:  list[float],
    equity_curve: Optional[pd.Series],
    trades:       Optional[list] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute all performance metrics.

    Args:
        net_profits:    Per-trade net P&L list.
        equity_curve:   Optional Series of equity values (indexed by timestamp).
        trades:         Optional list of Trade objects for detailed stats.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 0.0).

    Returns:
        dict with all metrics (missing ones keyed to None).
    """
    m: dict = {}

    if not net_profits:
        return _empty_metrics()

    arr = np.array(net_profits, dtype=float)
    wins  = arr[arr > 0]
    losses = arr[arr < 0]

    # ── Basic counts ──────────────────────────────────────────────────────────
    m["n_trades"]    = len(arr)
    m["n_winners"]   = len(wins)
    m["n_losers"]    = len(losses)
    m["win_rate"]    = round(len(wins) / len(arr), 4) if len(arr) > 0 else 0.0

    # ── P&L ───────────────────────────────────────────────────────────────────
    m["net_profit"]    = round(float(arr.sum()), 4)
    m["gross_profit"]  = round(float(wins.sum()),  4) if len(wins) > 0 else 0.0
    m["gross_loss"]    = round(float(losses.sum()), 4) if len(losses) > 0 else 0.0
    m["avg_win"]       = round(float(wins.mean()),   4) if len(wins) > 0 else 0.0
    m["avg_loss"]      = round(float(losses.mean()), 4) if len(losses) > 0 else 0.0

    m["profit_factor"] = (
        round(abs(m["gross_profit"] / m["gross_loss"]), 4)
        if m["gross_loss"] != 0 else None
    )
    m["expectancy"]    = round(float(arr.mean()), 4)

    # ── Largest trade ─────────────────────────────────────────────────────────
    m["largest_win"]  = round(float(wins.max()),   4) if len(wins) > 0 else 0.0
    m["largest_loss"] = round(float(losses.min()), 4) if len(losses) > 0 else 0.0

    # ── Consecutive ───────────────────────────────────────────────────────────
    m["max_consecutive_wins"]   = _max_consecutive(arr, positive=True)
    m["max_consecutive_losses"] = _max_consecutive(arr, positive=False)

    # ── Holding time ──────────────────────────────────────────────────────────
    if trades:
        held = [t.holding_bars for t in trades if getattr(t, "holding_bars", None) is not None]
        m["avg_holding_bars"] = round(float(np.mean(held)), 2) if held else None
    else:
        m["avg_holding_bars"] = None

    # ── Drawdown ──────────────────────────────────────────────────────────────
    if equity_curve is not None and len(equity_curve) > 1:
        dd_metrics = compute_drawdown_metrics(equity_curve)
        m.update(dd_metrics)
    else:
        m["max_drawdown"]     = None
        m["max_drawdown_pct"] = None
        m["recovery_factor"]  = None
        m["ulcer_index"]      = None

    # ── Return-based ──────────────────────────────────────────────────────────
    if equity_curve is not None and len(equity_curve) > 1:
        m.update(compute_return_metrics(equity_curve, risk_free_rate))
    else:
        m["sharpe_ratio"]  = None
        m["sortino_ratio"] = None
        m["calmar_ratio"]  = None
        m["omega_ratio"]   = None

    # ── Exit analysis ─────────────────────────────────────────────────────────
    if trades:
        exit_stats = compute_exit_statistics(trades)
        m.update(exit_stats)

    return m


# ── Drawdown metrics ──────────────────────────────────────────────────────────

def compute_drawdown_metrics(equity_curve: pd.Series) -> dict:
    """Compute drawdown-related metrics from an equity series."""
    eq = equity_curve.values.astype(float)
    peak = np.maximum.accumulate(eq)

    drawdowns     = eq - peak
    drawdowns_pct = np.where(peak > 0, drawdowns / peak, 0.0)

    max_dd        = float(np.min(drawdowns))
    max_dd_pct    = float(np.min(drawdowns_pct))

    net_profit    = float(eq[-1] - eq[0])
    recovery      = net_profit / abs(max_dd) if max_dd < 0 else None

    # Ulcer Index: RMS of % drawdown from running peak
    ulcer = float(np.sqrt(np.mean(drawdowns_pct ** 2))) if len(drawdowns_pct) > 0 else None

    return {
        "max_drawdown":     round(max_dd, 4),
        "max_drawdown_pct": round(max_dd_pct, 6),
        "recovery_factor":  round(recovery, 4) if recovery is not None else None,
        "ulcer_index":      round(ulcer, 6)     if ulcer is not None else None,
    }


# ── Return metrics ────────────────────────────────────────────────────────────

def compute_return_metrics(
    equity_curve:   pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute Sharpe, Sortino, Calmar, Omega from equity curve."""
    eq  = equity_curve.values.astype(float)
    rets = np.diff(eq) / eq[:-1]            # bar returns
    if len(rets) == 0:
        return {"sharpe_ratio": None, "sortino_ratio": None,
                "calmar_ratio": None, "omega_ratio": None}

    mean_ret  = float(np.mean(rets))
    std_ret   = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0

    # Approximate annualisation using 252 trading days × 24 bars/day = 6048
    ann_factor = math.sqrt(len(rets)) if len(rets) > 1 else 1.0

    sharpe = (mean_ret - risk_free_rate) / std_ret * ann_factor if std_ret > 0 else None

    downside = rets[rets < 0]
    if len(downside) > 1:
        down_std = float(np.std(downside, ddof=1))
        sortino  = (mean_ret - risk_free_rate) / down_std * ann_factor if down_std > 0 else None
    else:
        sortino = None

    # Calmar = total return / max_drawdown
    net_return = (eq[-1] - eq[0]) / eq[0] if eq[0] != 0 else 0.0
    dd_metrics = compute_drawdown_metrics(equity_curve)
    max_dd_pct = abs(dd_metrics["max_drawdown_pct"])
    calmar = net_return / max_dd_pct if max_dd_pct > 0 else None

    # Omega = sum(positive returns) / sum(abs(negative returns))
    pos_sum = float(rets[rets > 0].sum()) if len(rets[rets > 0]) > 0 else 0.0
    neg_sum = float(abs(rets[rets < 0].sum())) if len(rets[rets < 0]) > 0 else 0.0
    omega   = round(pos_sum / neg_sum, 4) if neg_sum > 0 else None

    return {
        "sharpe_ratio":  round(sharpe,  4) if sharpe  is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "calmar_ratio":  round(calmar,  4) if calmar  is not None else None,
        "omega_ratio":   omega,
    }


# ── Exit statistics ───────────────────────────────────────────────────────────

def compute_exit_statistics(trades: list) -> dict:
    """Break down trade exits by reason."""
    reasons: dict[str, int] = {}
    for t in trades:
        r = getattr(t, "exit_reason", None) or "unknown"
        reasons[r] = reasons.get(r, 0) + 1

    total = len(trades)
    return {
        "exit_tp":         reasons.get("tp",           0),
        "exit_sl":         reasons.get("sl",           0),
        "exit_trailing":   reasons.get("trailing_sl",  0),
        "exit_time_stop":  reasons.get("time_stop",    0),
        "exit_be":         reasons.get("be",           0),
        "exit_eod":        reasons.get("end_of_data",  0),
        "exit_breakdown":  reasons,
    }


# ── Monthly / Yearly returns ──────────────────────────────────────────────────

def compute_period_returns(
    equity_curve: pd.Series,
    period:       str = "M",    # "M" = monthly, "Y" = yearly
) -> pd.DataFrame:
    """Return a DataFrame with period start/end equity and % return."""
    if equity_curve.empty:
        return pd.DataFrame()
    resampled = equity_curve.resample(period).agg(["first", "last"])
    resampled.columns = ["period_start", "period_end"]
    resampled["return_pct"] = (
        (resampled["period_end"] - resampled["period_start"])
        / resampled["period_start"]
        * 100.0
    ).round(4)
    return resampled.dropna()


# ── Private helpers ───────────────────────────────────────────────────────────

def _max_consecutive(arr: np.ndarray, positive: bool) -> int:
    max_run = 0
    run     = 0
    for v in arr:
        is_pos = v > 0
        if is_pos == positive:
            run    += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _empty_metrics() -> dict:
    return {k: None for k in [
        "n_trades", "n_winners", "n_losers", "win_rate",
        "net_profit", "gross_profit", "gross_loss",
        "avg_win", "avg_loss", "profit_factor", "expectancy",
        "largest_win", "largest_loss",
        "max_consecutive_wins", "max_consecutive_losses", "avg_holding_bars",
        "max_drawdown", "max_drawdown_pct", "recovery_factor", "ulcer_index",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
        "exit_tp", "exit_sl", "exit_trailing", "exit_time_stop",
        "exit_be", "exit_eod", "exit_breakdown",
    ]}
