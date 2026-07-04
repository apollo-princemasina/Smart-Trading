"""
Execution Engine
================
Simulates realistic order execution costs at every trade entry and exit:
  - Spread  (fixed, applied as half-spread each side)
  - Commission (per round-trip lot)
  - Slippage (Gaussian noise around a mean)
  - Execution delay (N bars after the signal bar)
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ExecutionConfig:
    """All execution simulation parameters."""
    spread_pips:           float = 2.0      # full spread in pips
    commission_per_lot:    float = 7.0      # USD per standard lot round-trip
    slippage_pips:         float = 0.5      # mean slippage in pips
    slippage_std:          float = 0.3      # Gaussian sigma for slippage
    execution_delay_bars:  int   = 1        # bars between signal and entry
    pip_size:              float = 0.0001   # price value of 1 pip
    pip_value:             float = 10.0     # USD per pip per standard lot
    min_lot_size:          float = 0.01
    max_lot_size:          float = 100.0
    market_hours_only:     bool  = False    # unused — handled by analytics


class ExecutionEngine:
    """Apply execution costs to entries and exits."""

    def __init__(self, config: ExecutionConfig, random_seed: int = 42) -> None:
        self.cfg = config
        self._rng = random.Random(random_seed)

    # ── Entry ─────────────────────────────────────────────────────────────────

    def entry_bar(self, signal_bar_idx: int) -> int:
        """Bar index at which the trade is entered after the signal."""
        return signal_bar_idx + self.cfg.execution_delay_bars

    def calculate_entry(
        self,
        direction:  str,
        bar_open:   float,
    ) -> tuple[float, float, float]:
        """Compute (entry_price, spread_cost_usd, slippage_cost_usd).

        BUY:  entry = open + half_spread + slippage
        SELL: entry = open - half_spread - slippage
        """
        half_spread_px  = (self.cfg.spread_pips / 2.0) * self.cfg.pip_size
        slip_pips       = max(0.0, self._rng.gauss(self.cfg.slippage_pips, self.cfg.slippage_std))
        slip_px         = slip_pips * self.cfg.pip_size

        if direction == "BUY":
            entry_price = bar_open + half_spread_px + slip_px
        else:
            entry_price = bar_open - half_spread_px - slip_px

        spread_cost   = (half_spread_px / self.cfg.pip_size) * self.cfg.pip_value
        slippage_cost = (slip_px       / self.cfg.pip_size) * self.cfg.pip_value
        return round(entry_price, 5), round(spread_cost, 4), round(slippage_cost, 4)

    # ── Exit ──────────────────────────────────────────────────────────────────

    def calculate_exit(
        self,
        direction:     str,
        bar_price:     float,
        is_limit_exit: bool = False,
    ) -> float:
        """Return realistic exit price.

        Limit orders (TP/SL triggered at a known level) fill exactly.
        Market exits (end-of-data, time-stop) incur slippage.
        """
        if is_limit_exit:
            return round(bar_price, 5)

        slip_pips  = max(0.0, self._rng.gauss(self.cfg.slippage_pips, self.cfg.slippage_std))
        slip_px    = slip_pips * self.cfg.pip_size
        # Slippage works against the trader: BUY closed lower, SELL closed higher
        if direction == "BUY":
            return round(bar_price - slip_px, 5)
        return round(bar_price + slip_px, 5)

    # ── Commission ───────────────────────────────────────────────────────────

    def calculate_commission(self, lot_size: float) -> float:
        return round(lot_size * self.cfg.commission_per_lot, 4)

    # ── Lot clamping ─────────────────────────────────────────────────────────

    def clamp_lot(self, lot_size: float) -> float:
        return max(self.cfg.min_lot_size, min(self.cfg.max_lot_size, lot_size))
