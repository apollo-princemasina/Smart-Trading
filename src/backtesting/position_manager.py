"""
Position Manager
================
Calculates lot sizes for new trades.

Supported sizing modes
----------------------
  fixed_lot      — always the same lot size
  fixed_risk_pct — risk a fixed % of current balance per trade
  atr_sizing     — normalise risk to 1× ATR volatility unit
  kelly          — Kelly Criterion based on model's estimated win-rate/RR
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PositionConfig:
    """Lot-sizing configuration."""
    mode:             str   = "fixed_lot"   # "fixed_lot"|"fixed_risk_pct"|"atr_sizing"|"kelly"

    # ── Fixed lot ────────────────────────────────────────────────────────────
    fixed_lot_size:   float = 0.10

    # ── Fixed risk % ─────────────────────────────────────────────────────────
    risk_pct:         float = 0.01          # 1 % of balance per trade

    # ── ATR sizing ───────────────────────────────────────────────────────────
    atr_risk_pct:     float = 0.01
    atr_column:       str   = "atr"

    # ── Kelly ────────────────────────────────────────────────────────────────
    kelly_fraction:   float = 0.25          # conservative quarter-Kelly
    kelly_win_rate:   float = 0.55          # fallback if not enough trades
    kelly_avg_win:    float = 2.0           # average win in RR units
    kelly_avg_loss:   float = 1.0           # average loss in RR units

    # ── Common limits ────────────────────────────────────────────────────────
    pip_size:         float = 0.0001
    pip_value:        float = 10.0          # USD/pip/lot
    min_lot:          float = 0.01
    max_lot:          float = 10.0


class PositionManager:
    """Determine lot size for each trade signal."""

    def __init__(self, config: PositionConfig) -> None:
        self.cfg = config
        self._win_history:  list[bool]  = []
        self._rr_history:   list[float] = []   # realised RR for Kelly update

    def compute_lot(
        self,
        balance:      float,
        direction:    str,
        entry_price:  float,
        stop_loss:    Optional[float],
        bar_idx:      int,
        price_df:     "pd.DataFrame",
        confidence:   float = 0.5,
    ) -> float:
        """Return lot size (clamped to [min_lot, max_lot]).

        Args:
            balance:     Current account balance.
            direction:   "BUY" or "SELL".
            entry_price: Trade entry price.
            stop_loss:   SL price (needed for risk-based sizing).
            bar_idx:     Bar index for ATR lookup.
            price_df:    Full price DataFrame.
            confidence:  Model probability (used by Kelly variant).
        """
        mode = self.cfg.mode

        if mode == "fixed_lot":
            lot = self.cfg.fixed_lot_size

        elif mode == "fixed_risk_pct":
            lot = self._fixed_risk_lot(balance, entry_price, stop_loss)

        elif mode == "atr_sizing":
            lot = self._atr_lot(balance, bar_idx, price_df)

        elif mode == "kelly":
            lot = self._kelly_lot(balance, confidence)

        else:
            lot = self.cfg.fixed_lot_size

        return self._clamp(lot)

    def record_trade_outcome(self, is_winner: bool, realised_rr: float) -> None:
        """Feed completed trade results back for Kelly updates."""
        self._win_history.append(is_winner)
        self._rr_history.append(realised_rr)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fixed_risk_lot(
        self,
        balance:     float,
        entry_price: float,
        stop_loss:   Optional[float],
    ) -> float:
        if stop_loss is None or stop_loss == entry_price:
            return self.cfg.fixed_lot_size
        risk_usd  = balance * self.cfg.risk_pct
        sl_pips   = abs(entry_price - stop_loss) / self.cfg.pip_size
        if sl_pips < 1:
            return self.cfg.min_lot
        lot = risk_usd / (sl_pips * self.cfg.pip_value)
        return lot

    def _atr_lot(self, balance: float, bar_idx: int, price_df: "pd.DataFrame") -> float:
        atr = self._get_atr(price_df, bar_idx)
        if atr <= 0:
            return self.cfg.fixed_lot_size
        atr_pips = atr / self.cfg.pip_size
        risk_usd = balance * self.cfg.atr_risk_pct
        lot      = risk_usd / (atr_pips * self.cfg.pip_value)
        return lot

    def _kelly_lot(self, balance: float, confidence: float) -> float:
        if len(self._win_history) >= 20:
            win_rate = sum(self._win_history) / len(self._win_history)
            avg_win  = np.mean([r for r in self._rr_history if r > 0]) if any(r > 0 for r in self._rr_history) else self.cfg.kelly_avg_win
            avg_loss = abs(np.mean([r for r in self._rr_history if r < 0])) if any(r < 0 for r in self._rr_history) else self.cfg.kelly_avg_loss
        else:
            win_rate = confidence          # use model confidence as prior
            avg_win  = self.cfg.kelly_avg_win
            avg_loss = self.cfg.kelly_avg_loss

        if avg_loss == 0:
            return self.cfg.fixed_lot_size

        b      = avg_win / avg_loss        # reward-to-risk ratio
        kelly  = (b * win_rate - (1 - win_rate)) / b
        kelly  = max(0.0, kelly) * self.cfg.kelly_fraction
        lot    = balance * kelly / (self.cfg.kelly_avg_loss * self.cfg.pip_value * 20)
        return lot

    def _get_atr(self, price_df: "pd.DataFrame", bar_idx: int) -> float:
        col = self.cfg.atr_column
        if col in price_df.columns and bar_idx < len(price_df):
            val = price_df[col].iloc[bar_idx]
            if val and val > 0:
                return float(val)
        return self.cfg.pip_size * 20   # fallback: 20 pips

    def _clamp(self, lot: float) -> float:
        return round(max(self.cfg.min_lot, min(self.cfg.max_lot, lot)), 2)
