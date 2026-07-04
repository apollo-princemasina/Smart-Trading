"""Session-aware confidence weighting.

ICT killzones define windows of high institutional participation.
Signals generated outside these windows carry lower conviction regardless
of what the model outputs — the model was trained on all hours equally
but live trading is most reliable during active sessions.

Multipliers are applied to the raw directional probability. The residual
probability is redistributed to HOLD. If adjusted confidence falls below
INFERENCE_MIN_CONFIDENCE, the signal is demoted to HOLD (no trade).
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionInfo:
    name: str          # LONDON_OPEN | NY_OPEN | ASIAN | LONDON_CLOSE | DEAD_ZONE | MARKET_CLOSED
    multiplier: float  # applied to raw directional probability
    active: bool       # True = inside a killzone


# Multipliers chosen conservatively:
#   London/NY open: full model conviction (these sessions generated most training signals)
#   Asian: slightly reduced (EUR/USD range-building, direction less reliable)
#   London Close: moderate (often mean-reversion, can trap directional trades)
#   Dead Zone: significant reduction (low volume, false moves, spread widens)
_SESSIONS = [
    # (utc_start_inclusive, utc_end_exclusive, name, multiplier, active)
    (7,  10, "LONDON_OPEN",   1.00, True),
    (12, 15, "NY_OPEN",       1.00, True),
    (0,  4,  "ASIAN",         0.80, True),
    (15, 17, "LONDON_CLOSE",  0.82, True),
    # Dead zones — everything else
]


_MARKET_CLOSED = SessionInfo(name="MARKET_CLOSED", multiplier=0.0, active=False)


def _is_market_closed(now: datetime) -> bool:
    """Forex is closed from Friday 22:00 UTC to Sunday 22:00 UTC."""
    weekday = now.weekday()  # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    hour = now.hour
    return (
        (weekday == 4 and hour >= 22)   # Friday after 22:00
        or weekday == 5                  # All of Saturday
        or (weekday == 6 and hour < 22)  # Sunday before 22:00
    )


def get_session(utc_hour: int | None = None) -> SessionInfo:
    now = datetime.now(timezone.utc)
    if _is_market_closed(now):
        return _MARKET_CLOSED
    if utc_hour is None:
        utc_hour = now.hour
    for start, end, name, mult, active in _SESSIONS:
        if start <= utc_hour < end:
            return SessionInfo(name=name, multiplier=mult, active=active)
    return SessionInfo(name="DEAD_ZONE", multiplier=0.60, active=False)


def apply_session_weighting(
    prob_sell: float,
    prob_hold: float,
    prob_buy: float,
    direction: str,
    session: SessionInfo,
) -> tuple[float, float, float, float]:
    """Apply session multiplier to directional probability, redistribute to HOLD.

    Returns: (adj_prob_sell, adj_prob_hold, adj_prob_buy, adj_confidence)
    """
    mult = session.multiplier

    if direction == "BUY":
        adj_buy  = prob_buy * mult
        spill    = prob_buy - adj_buy          # redistributed to HOLD
        adj_sell = prob_sell
        adj_hold = min(prob_hold + spill, 1.0)
        adj_conf = adj_buy
    elif direction == "SELL":
        adj_sell = prob_sell * mult
        spill    = prob_sell - adj_sell
        adj_buy  = prob_buy
        adj_hold = min(prob_hold + spill, 1.0)
        adj_conf = adj_sell
    else:  # HOLD — session weighting doesn't change holds
        adj_sell, adj_hold, adj_buy = prob_sell, prob_hold, prob_buy
        adj_conf = prob_hold

    # Re-normalise so probabilities sum to 1
    total = adj_sell + adj_hold + adj_buy
    if total > 0:
        adj_sell /= total
        adj_hold /= total
        adj_buy  /= total

    return round(adj_sell, 4), round(adj_hold, 4), round(adj_buy, 4), round(adj_conf, 4)
