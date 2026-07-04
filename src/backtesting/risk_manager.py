"""
Risk Manager
============
Guards the portfolio against over-trading and excessive drawdown:

  - Maximum simultaneously open positions
  - Maximum daily loss (flat for the day once hit)
  - Maximum weekly loss
  - Minimum confidence threshold (second line of defence after TradeEngine)
  - Correlation guard (no opposing BUY+SELL open at the same time)

The risk manager never modifies model predictions; it only decides
whether to ACCEPT or REJECT a proposed TradeSignal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_open_positions:     int   = 3
    max_daily_loss_pct:     float = 0.02    # 2 % of initial capital
    max_weekly_loss_pct:    float = 0.05    # 5 % of initial capital
    min_confidence:         float = 0.60    # redundant guard (TradeEngine also filters)
    allow_simultaneous_rr:  bool  = False   # allow both BUY and SELL open at once
    initial_capital:        float = 10_000.0


class RiskManager:
    """Evaluate risk conditions before opening a new trade.

    Methods return (allowed: bool, reason: str).
    """

    def __init__(self, config: RiskConfig) -> None:
        self.cfg             = config
        self._daily_loss:   dict[str, float] = {}    # date → cumulative loss
        self._weekly_loss:  dict[str, float] = {}    # week → cumulative loss
        self._flat_days:    set[str]          = set()
        self._flat_weeks:   set[str]          = set()

    def check(
        self,
        timestamp:       pd.Timestamp,
        direction:       str,
        confidence:      float,
        n_open_positions: int,
        open_directions: list[str],     # directions of currently open trades
    ) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        day_key  = timestamp.strftime("%Y-%m-%d")
        week_key = timestamp.strftime("%Y-W%W")

        if day_key in self._flat_days:
            return False, "daily_loss_limit"

        if week_key in self._flat_weeks:
            return False, "weekly_loss_limit"

        if n_open_positions >= self.cfg.max_open_positions:
            return False, "max_positions"

        if confidence < self.cfg.min_confidence:
            return False, "low_confidence"

        if not self.cfg.allow_simultaneous_rr:
            opp = "SELL" if direction == "BUY" else "BUY"
            if opp in open_directions:
                return False, "opposing_position"

        return True, "ok"

    def record_closed_trade(
        self,
        timestamp:  pd.Timestamp,
        net_profit: float,
    ) -> None:
        """Update running P&L for daily/weekly drawdown tracking."""
        if net_profit >= 0:
            return   # only track losses

        day_key  = timestamp.strftime("%Y-%m-%d")
        week_key = timestamp.strftime("%Y-W%W")

        self._daily_loss[day_key]   = self._daily_loss.get(day_key,  0.0) + abs(net_profit)
        self._weekly_loss[week_key] = self._weekly_loss.get(week_key, 0.0) + abs(net_profit)

        cap = self.cfg.initial_capital
        if self._daily_loss[day_key]  >= cap * self.cfg.max_daily_loss_pct:
            self._flat_days.add(day_key)

        if self._weekly_loss[week_key] >= cap * self.cfg.max_weekly_loss_pct:
            self._flat_weeks.add(week_key)

    def reset_daily(self, date_str: str) -> None:
        """Remove a day from the flat-days set (testing helper)."""
        self._flat_days.discard(date_str)

    @property
    def flat_days(self) -> set[str]:
        return set(self._flat_days)

    @property
    def flat_weeks(self) -> set[str]:
        return set(self._flat_weeks)
