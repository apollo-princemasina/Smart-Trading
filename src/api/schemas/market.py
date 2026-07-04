from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class CandleOut(BaseModel):
    timestamp: datetime
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    int


class CandleBufferOut(BaseModel):
    symbol:    str
    timeframe: str
    count:     int
    candles:   list[CandleOut]


class RegimeScores(BaseModel):
    consolidation: float
    expansion:     float
    manipulation:  float


class ICTSignals(BaseModel):
    liquidity_sweep:  bool
    sweep_direction:  str     # BULLISH | BEARISH | NONE
    sweep_rejected:   bool
    sweep_confirmed:  bool
    choch_detected:   bool
    choch_direction:  str
    bos_detected:     bool
    bos_direction:    str
    fvg_active:       bool
    fvg_direction:    str
    ob_active:        bool
    ob_direction:     str
    in_order_block:   bool


class RegimeOut(BaseModel):
    symbol:         str
    timeframe:      str
    timestamp:      Optional[datetime] = None
    dominant:       str               # CONSOLIDATION | EXPANSION | MANIPULATION
    scores:         RegimeScores
    bias:           str               # BULLISH | BEARISH | NEUTRAL
    pd_zone:        str               # PREMIUM | DISCOUNT | EQUILIBRIUM
    atr_pips:       Optional[float]   = None
    atr_vs_avg:     Optional[str]     = None
    adx:            Optional[float]   = None
    ict:            ICTSignals
    narrative:      str
    trade_implication: str
