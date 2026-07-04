from __future__ import annotations
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel


class PredictionOut(BaseModel):
    id:              UUID
    created_at:      datetime
    signal_time:     datetime
    symbol:          str
    timeframe:       str
    direction:       str          # BUY | SELL | HOLD
    confidence:      float        # session-adjusted
    raw_confidence:  Optional[float] = None  # raw model output before weighting
    prob_sell:       float
    prob_hold:       float
    prob_buy:        float
    close:           float
    atr_pips:        Optional[float] = None
    tp_price:        Optional[float] = None
    sl_price:        Optional[float] = None
    tp_pips:         Optional[float] = None
    sl_pips:         Optional[float] = None
    regime:          Optional[str]   = None
    session:         Optional[str]   = None  # LONDON_OPEN | NY_OPEN | ASIAN | LONDON_CLOSE | DEAD_ZONE
    session_mult:    Optional[float] = None  # multiplier applied (0.60–1.0)
    model_version:   Optional[str]   = None

    model_config = {"from_attributes": True}


class PredictionListOut(BaseModel):
    total:       int
    page:        int
    page_size:   int
    predictions: list[PredictionOut]


class OutcomeOut(BaseModel):
    id:            UUID
    prediction_id: UUID
    evaluated_at:  datetime
    outcome:       str          # TP_HIT | SL_HIT | EXPIRED | PENDING
    exit_price:    Optional[float] = None
    pnl_pips:      Optional[float] = None
    bars_to_exit:  Optional[int]   = None

    model_config = {"from_attributes": True}
