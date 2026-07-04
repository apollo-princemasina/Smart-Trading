"""
Backtesting performance metrics — re-exports from performance.py.

This module is kept for backward compatibility with any code that imports
from src.backtesting.metrics.  All real implementations live in performance.py.
"""
from .performance import (
    compute_metrics,
    compute_performance_metrics,
    compute_drawdown_metrics,
    compute_return_metrics,
    compute_exit_statistics,
    compute_period_returns,
)

__all__ = [
    "compute_metrics",
    "compute_performance_metrics",
    "compute_drawdown_metrics",
    "compute_return_metrics",
    "compute_exit_statistics",
    "compute_period_returns",
]
