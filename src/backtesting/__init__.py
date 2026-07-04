"""
src.backtesting — Institutional AI Backtesting Engine
=====================================================
Evaluates ML model predictions as if they were traded in real time.

Quick start
-----------
    from src.backtesting import Backtester, BacktestConfig
    from src.backtesting.execution_engine import ExecutionConfig
    from src.backtesting.sl_tp_manager import SLTPConfig
    from src.backtesting.position_manager import PositionConfig

    cfg = BacktestConfig(
        bundle_dir      = Path("models/best_model/bundle"),
        output_dir      = Path("backtesting"),
        initial_capital = 10_000.0,
        min_probability = 0.65,
        sl_tp           = SLTPConfig(mode="atr", sl_atr_mult=1.5, tp_atr_mult=3.0),
        position        = PositionConfig(mode="fixed_risk_pct", risk_pct=0.01),
    )
    result = Backtester(cfg).run(price_df=df)
"""
from .backtester import Backtester, BacktestConfig, BacktestResult, run_backtest
from .portfolio import Portfolio, Trade, EquitySnapshot
from .trade_engine import TradeEngine, TradeSignal
from .execution_engine import ExecutionEngine, ExecutionConfig
from .sl_tp_manager import SLTPManager, SLTPConfig
from .position_manager import PositionManager, PositionConfig
from .risk_manager import RiskManager, RiskConfig
from .performance import (
    compute_metrics,
    compute_performance_metrics,
    compute_drawdown_metrics,
    compute_return_metrics,
    compute_exit_statistics,
    compute_period_returns,
)
from .analytics import (
    get_session,
    classify_regime,
    analyze_confidence_bands,
    analyze_session_performance,
    analyze_direction_performance,
    analyze_regime_performance,
    compute_rolling_accuracy,
)
from .reports import BacktestReporter, create_report

__all__ = [
    # Orchestrator
    "Backtester", "BacktestConfig", "BacktestResult", "run_backtest",
    # Portfolio
    "Portfolio", "Trade", "EquitySnapshot",
    # Signal engine
    "TradeEngine", "TradeSignal",
    # Execution
    "ExecutionEngine", "ExecutionConfig",
    # SL/TP
    "SLTPManager", "SLTPConfig",
    # Position sizing
    "PositionManager", "PositionConfig",
    # Risk
    "RiskManager", "RiskConfig",
    # Performance
    "compute_metrics", "compute_performance_metrics",
    "compute_drawdown_metrics", "compute_return_metrics",
    "compute_exit_statistics", "compute_period_returns",
    # Analytics
    "get_session", "classify_regime",
    "analyze_confidence_bands", "analyze_session_performance",
    "analyze_direction_performance", "analyze_regime_performance",
    "compute_rolling_accuracy",
    # Reports
    "BacktestReporter", "create_report",
]
