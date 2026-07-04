"""
Strategy definitions — re-exports from trade_engine.py.

Legacy shim; all logic is in TradeEngine.
"""
from .trade_engine import TradeEngine, TradeSignal


def select_trades(signals: list) -> list:
    """Return all actionable signals.  Legacy shim."""
    return [s for s in signals if getattr(s, "is_actionable", False)]
