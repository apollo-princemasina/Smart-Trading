"""
Portfolio
=========
Tracks account balance, open positions, closed trades, and equity history.

Design constraints
------------------
  Read-only for predictions — never calls any model.
  Never modifies config parameters during the run.
  Equity history is recorded at every bar for smooth curve generation.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ── Trade dataclass ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Complete record of one backtested trade.

    This is the central data structure of the backtesting engine.  Every
    attribute is set once (immutable after closing).
    """
    trade_id:              str
    direction:             str               # "BUY" or "SELL"
    signal_time:           pd.Timestamp
    entry_time:            Optional[pd.Timestamp] = None
    exit_time:             Optional[pd.Timestamp] = None
    entry_price:           float             = 0.0
    exit_price:            Optional[float]   = None
    stop_loss:             Optional[float]   = None
    take_profit:           Optional[float]   = None
    trailing_sl:           Optional[float]   = None  # current trailing SL level
    be_activated:          bool              = False  # break-even activated
    lot_size:              float             = 0.0
    commission:            float             = 0.0   # total commission cost
    spread_cost:           float             = 0.0   # cost of spread
    slippage_cost:         float             = 0.0   # cost of slippage
    confidence:            float             = 0.0   # model prediction probability
    prediction_class:      int               = 0
    status:                str               = "pending"  # "pending"|"open"|"closed"|"rejected"
    profit_pips:           Optional[float]   = None
    gross_profit:          Optional[float]   = None  # before transaction costs
    net_profit:            Optional[float]   = None  # after all costs
    exit_reason:           Optional[str]     = None  # "tp"|"sl"|"time_stop"|"trailing_sl"|"be"|"end_of_data"
    holding_bars:          Optional[int]     = None
    entry_bar_idx:         Optional[int]     = None  # bar index at entry
    atr_at_entry:          Optional[float]   = None
    session:               Optional[str]     = None  # "london"|"newyork"|"asian"|"overlap"|"offhours"
    features_snapshot:     Optional[dict]    = field(default=None, repr=False)

    @property
    def is_winner(self) -> Optional[bool]:
        if self.net_profit is None:
            return None
        return self.net_profit > 0

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def unrealized_pnl(
        self, current_price: float, pip_size: float, pip_value: float
    ) -> float:
        """Mark-to-market unrealized PnL at current_price."""
        if self.status != "open":
            return 0.0
        mult = 1.0 if self.direction == "BUY" else -1.0
        pips = mult * (current_price - self.entry_price) / pip_size
        return pips * pip_value * self.lot_size - self.commission


# ── Equity snapshot ────────────────────────────────────────────────────────────

@dataclass
class EquitySnapshot:
    timestamp:    pd.Timestamp
    bar_idx:      int
    balance:      float    # realised P&L only
    equity:       float    # balance + unrealised P&L
    drawdown:     float    # equity - peak_equity (non-positive)
    drawdown_pct: float    # drawdown / peak_equity (non-positive)
    open_trades:  int


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:
    """Maintains account state throughout the backtest."""

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.initial_capital: float = initial_capital
        self.balance:         float = initial_capital
        self._peak_equity:    float = initial_capital
        self._open: dict[str, Trade]   = {}
        self._closed: list[Trade]      = []
        self._equity_history: list[EquitySnapshot] = []

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self.balance  # updated by record_equity()

    @property
    def open_trades(self) -> list[Trade]:
        return list(self._open.values())

    @property
    def closed_trades(self) -> list[Trade]:
        return list(self._closed)

    @property
    def all_trades(self) -> list[Trade]:
        return self._closed + list(self._open.values())

    @property
    def equity_history(self) -> list[EquitySnapshot]:
        return list(self._equity_history)

    # ── Trade management ───────────────────────────────────────────────────────

    def open_trade(self, trade: Trade) -> None:
        """Register a trade as open.  Deducts commission from balance."""
        trade.status = "open"
        self.balance -= trade.commission
        self._open[trade.trade_id] = trade

    def close_trade(
        self,
        trade:        Trade,
        exit_price:   float,
        exit_reason:  str,
        exit_time:    pd.Timestamp,
        exit_bar_idx: int,
        pip_size:     float = 0.0001,
        pip_value:    float = 10.0,
    ) -> Trade:
        """Close an open trade and update the balance."""
        if trade.trade_id not in self._open:
            return trade

        mult           = 1.0 if trade.direction == "BUY" else -1.0
        profit_pips    = mult * (exit_price - trade.entry_price) / pip_size
        gross_profit   = profit_pips * pip_value * trade.lot_size
        net_profit     = gross_profit - trade.spread_cost - trade.slippage_cost

        holding = (
            exit_bar_idx - trade.entry_bar_idx
            if trade.entry_bar_idx is not None else None
        )

        trade.exit_price     = exit_price
        trade.exit_time      = exit_time
        trade.exit_reason    = exit_reason
        trade.status         = "closed"
        trade.profit_pips    = round(profit_pips, 4)
        trade.gross_profit   = round(gross_profit, 4)
        trade.net_profit     = round(net_profit, 4)
        trade.holding_bars   = holding

        self.balance += net_profit
        del self._open[trade.trade_id]
        self._closed.append(trade)
        return trade

    def reject_trade(self, trade: Trade, reason: str = "risk_limit") -> None:
        trade.status      = "rejected"
        trade.exit_reason = reason

    # ── Equity recording ───────────────────────────────────────────────────────

    def record_equity(
        self,
        timestamp:     pd.Timestamp,
        bar_idx:       int,
        current_price: float,
        pip_size:      float = 0.0001,
        pip_value:     float = 10.0,
    ) -> None:
        """Snapshot equity (balance + unrealised PnL) at this bar."""
        unrealised = sum(
            t.unrealized_pnl(current_price, pip_size, pip_value)
            for t in self._open.values()
        )
        equity       = self.balance + unrealised
        self._peak_equity = max(self._peak_equity, equity)
        drawdown     = equity - self._peak_equity
        dd_pct       = drawdown / self._peak_equity if self._peak_equity > 0 else 0.0

        self._equity_history.append(EquitySnapshot(
            timestamp   = timestamp,
            bar_idx     = bar_idx,
            balance     = round(self.balance, 4),
            equity      = round(equity, 4),
            drawdown    = round(drawdown, 4),
            drawdown_pct = round(dd_pct, 6),
            open_trades = len(self._open),
        ))

    def equity_dataframe(self) -> pd.DataFrame:
        """Return equity history as a DataFrame indexed by timestamp."""
        if not self._equity_history:
            return pd.DataFrame()
        rows = [
            {
                "timestamp":    s.timestamp,
                "balance":      s.balance,
                "equity":       s.equity,
                "drawdown":     s.drawdown,
                "drawdown_pct": s.drawdown_pct,
                "open_trades":  s.open_trades,
            }
            for s in self._equity_history
        ]
        df = pd.DataFrame(rows).set_index("timestamp")
        return df

    # ── Convenience ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        closed = self._closed
        winners = [t for t in closed if t.net_profit and t.net_profit > 0]
        return {
            "initial_capital":  self.initial_capital,
            "final_balance":    round(self.balance, 4),
            "n_closed_trades":  len(closed),
            "n_open_trades":    len(self._open),
            "n_winners":        len(winners),
            "n_losers":         len(closed) - len(winners),
        }
