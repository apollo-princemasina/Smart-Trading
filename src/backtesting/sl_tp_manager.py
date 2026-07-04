"""
SL/TP Manager
=============
Calculates and manages stop-loss and take-profit levels throughout a trade's life.

Supported modes
---------------
  fixed_pips   — constant SL/TP distance in pips from entry
  atr          — SL/TP as multiples of the bar's ATR
  ict_dynamic  — ICT-style: SL beyond swing high/low, TP at next HTF level
  trailing     — SL trails price at a fixed pip distance
  break_even   — SL moved to break-even once reward reaches a threshold
  time_stop    — forcibly close after N bars

Multiple exit modes can be active simultaneously (trailing + break-even is common).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SLTPConfig:
    """Configuration for a complete SL/TP strategy."""
    mode:               str   = "fixed_pips"  # "fixed_pips"|"atr"|"ict_dynamic"|"trailing"

    # ── Fixed pips ────────────────────────────────────────────────────────────
    sl_pips:            float = 20.0
    tp_pips:            float = 40.0

    # ── ATR ───────────────────────────────────────────────────────────────────
    sl_atr_mult:        float = 1.5
    tp_atr_mult:        float = 3.0
    atr_period:         int   = 14
    atr_column:         str   = "atr"    # column name in price DataFrame

    # ── ICT Dynamic ──────────────────────────────────────────────────────────
    ict_swing_lookback: int   = 5        # bars to look back for swing high/low
    ict_buffer_pips:    float = 2.0      # extra buffer beyond swing

    # ── Trailing Stop ────────────────────────────────────────────────────────
    enable_trailing:    bool  = False
    trailing_pips:      float = 20.0     # distance from current price

    # ── Break-Even ───────────────────────────────────────────────────────────
    enable_break_even:  bool  = False
    be_trigger_rr:      float = 1.0      # move SL to entry after RR = be_trigger_rr
    be_buffer_pips:     float = 2.0      # SL = entry + buffer

    # ── Time Stop ────────────────────────────────────────────────────────────
    enable_time_stop:   bool  = False
    max_holding_bars:   int   = 48       # max bars before forced close

    # ── Pip info ─────────────────────────────────────────────────────────────
    pip_size:           float = 0.0001


class SLTPManager:
    """Compute initial SL/TP levels and update them bar-by-bar."""

    def __init__(self, config: SLTPConfig) -> None:
        self.cfg = config

    # ── Initial level calculation ─────────────────────────────────────────────

    def compute_initial_levels(
        self,
        direction:   str,
        entry_price: float,
        bar_idx:     int,
        price_df:    "pd.DataFrame",
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (stop_loss, take_profit) after entry.

        Args:
            direction:   "BUY" or "SELL".
            entry_price: Filled entry price.
            bar_idx:     Index of the entry bar in price_df.
            price_df:    Full price DataFrame (needed for ATR/ICT).

        Returns:
            (stop_loss, take_profit) — either may be None if not used.
        """
        mode = self.cfg.mode
        ps   = self.cfg.pip_size
        mult = 1.0 if direction == "BUY" else -1.0

        if mode == "fixed_pips":
            sl = entry_price - mult * self.cfg.sl_pips * ps
            tp = entry_price + mult * self.cfg.tp_pips * ps

        elif mode == "atr":
            atr = self._get_atr(price_df, bar_idx)
            sl  = entry_price - mult * self.cfg.sl_atr_mult * atr
            tp  = entry_price + mult * self.cfg.tp_atr_mult * atr

        elif mode == "ict_dynamic":
            sl, tp = self._ict_levels(direction, entry_price, bar_idx, price_df)

        else:
            # Fallback to fixed pips
            sl = entry_price - mult * self.cfg.sl_pips * ps
            tp = entry_price + mult * self.cfg.tp_pips * ps

        return round(sl, 5), round(tp, 5)

    # ── Bar-by-bar management ─────────────────────────────────────────────────

    def update(
        self,
        direction:     str,
        entry_price:   float,
        current_price: float,
        current_sl:    Optional[float],
        current_tp:    Optional[float],
        be_activated:  bool,
        bars_held:     int,
    ) -> tuple[Optional[float], Optional[float], bool, Optional[str]]:
        """Update SL/TP each bar.  Returns (new_sl, new_tp, be_activated, exit_reason).

        exit_reason is non-None when a time-stop triggers.
        """
        new_sl         = current_sl
        new_tp         = current_tp
        exit_reason: Optional[str] = None
        ps             = self.cfg.pip_size
        mult           = 1.0 if direction == "BUY" else -1.0

        # ── Trailing stop ─────────────────────────────────────────────────────
        if self.cfg.enable_trailing and current_sl is not None:
            trail_level = current_price - mult * self.cfg.trailing_pips * ps
            if mult * trail_level > mult * current_sl:
                new_sl = round(trail_level, 5)

        # ── Break-even ────────────────────────────────────────────────────────
        if self.cfg.enable_break_even and not be_activated and current_sl is not None:
            profit_pips = mult * (current_price - entry_price) / ps
            if current_tp is not None:
                tp_pips = mult * (current_tp - entry_price) / ps
            else:
                tp_pips = self.cfg.tp_pips

            if tp_pips > 0 and profit_pips / tp_pips >= self.cfg.be_trigger_rr:
                be_level = entry_price + mult * self.cfg.be_buffer_pips * ps
                if mult * be_level > mult * current_sl:
                    new_sl        = round(be_level, 5)
                    be_activated  = True

        # ── Time stop ─────────────────────────────────────────────────────────
        if self.cfg.enable_time_stop and bars_held >= self.cfg.max_holding_bars:
            exit_reason = "time_stop"

        return new_sl, new_tp, be_activated, exit_reason

    # ── Exit checks ───────────────────────────────────────────────────────────

    def check_sl_tp_hit(
        self,
        direction:  str,
        bar_high:   float,
        bar_low:    float,
        stop_loss:  Optional[float],
        take_profit: Optional[float],
    ) -> Optional[tuple[str, float]]:
        """Check if SL or TP was hit during this bar.

        Returns ("sl", price) | ("tp", price) | None.
        On the same bar both hit, SL takes priority (conservative).
        """
        sl_hit = tp_hit = False

        if stop_loss is not None:
            if direction == "BUY" and bar_low  <= stop_loss:  sl_hit = True
            if direction == "SELL" and bar_high >= stop_loss: sl_hit = True

        if take_profit is not None:
            if direction == "BUY" and bar_high >= take_profit:  tp_hit = True
            if direction == "SELL" and bar_low  <= take_profit: tp_hit = True

        if sl_hit:
            return ("sl", stop_loss)
        if tp_hit:
            return ("tp", take_profit)
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_atr(self, price_df: "pd.DataFrame", bar_idx: int) -> float:
        col = self.cfg.atr_column
        if col in price_df.columns and bar_idx < len(price_df):
            val = price_df[col].iloc[bar_idx]
            if val and val > 0:
                return float(val)
        return self.cfg.sl_pips * self.cfg.pip_size  # fallback

    def _ict_levels(
        self,
        direction:   str,
        entry_price: float,
        bar_idx:     int,
        price_df:    "pd.DataFrame",
    ) -> tuple[float, float]:
        lb   = self.cfg.ict_swing_lookback
        ps   = self.cfg.pip_size
        buf  = self.cfg.ict_buffer_pips * ps
        mult = 1.0 if direction == "BUY" else -1.0

        start = max(0, bar_idx - lb)
        window = price_df.iloc[start: bar_idx + 1]

        if "high" in window.columns and "low" in window.columns:
            swing_high = float(window["high"].max())
            swing_low  = float(window["low"].min())
        else:
            swing_high = entry_price + self.cfg.sl_pips * ps
            swing_low  = entry_price - self.cfg.sl_pips * ps

        if direction == "BUY":
            sl = swing_low  - buf
            tp = swing_high + buf
        else:
            sl = swing_high + buf
            tp = swing_low  - buf

        return round(sl, 5), round(tp, 5)
