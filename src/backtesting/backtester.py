"""
Backtester
==========
Main orchestrator for the Institutional AI Backtesting Engine.

Data flow
---------
  1.  Load price DataFrame (OHLCV + feature columns)
  2.  Load InferencePipeline from an optimized/validated bundle
  3.  Generate predictions for all bars (no retraining, no param changes)
  4.  TradeEngine converts predictions → TradeSignal list
  5.  For each bar (in chronological order):
        a. SLTPManager updates SL/TP for open trades
        b. SLTPManager checks if any open trade's SL/TP was hit
        c. RiskManager evaluates new signals
        d. ExecutionEngine fills pending entries at bar-open
        e. Portfolio records equity snapshot
  6.  Close any remaining open trades at the last bar's close
  7.  PerformanceEngine computes 20+ metrics
  8.  Analytics module annotates trades with session / regime
  9.  BacktestReporter writes 8 output files

Prediction guarantee
--------------------
  The backtester NEVER calls model.fit(), NEVER modifies predictions, and
  NEVER changes BacktestConfig after initialization.  Predictions are
  generated once in step 3 and treated as immutable thereafter.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .analytics import (
    analyze_confidence_bands,
    analyze_direction_performance,
    analyze_regime_performance,
    analyze_session_performance,
    get_session,
)
from .execution_engine import ExecutionConfig, ExecutionEngine
from .performance import compute_performance_metrics
from .portfolio import Portfolio, Trade
from .position_manager import PositionConfig, PositionManager
from .reports import BacktestReporter
from .risk_manager import RiskConfig, RiskManager
from .sl_tp_manager import SLTPConfig, SLTPManager
from .trade_engine import TradeEngine, TradeSignal
from src.optimization.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Complete backtest configuration.  Immutable after creation."""
    # ── Paths ─────────────────────────────────────────────────────────────────
    bundle_dir:       Path              # trained model bundle
    price_data_path:  Optional[Path] = None   # if None, price_df must be passed directly
    output_dir:       Path = Path("backtesting")

    # ── Target & features ─────────────────────────────────────────────────────
    target_column:    str  = "direction_1b"
    timestamp_column: str  = "time"         # OHLCV timestamp column

    # ── Signal filtering ──────────────────────────────────────────────────────
    min_probability:  float = 0.60

    # ── Execution ─────────────────────────────────────────────────────────────
    execution:        ExecutionConfig = field(default_factory=ExecutionConfig)

    # ── Position sizing ───────────────────────────────────────────────────────
    position:         PositionConfig  = field(default_factory=PositionConfig)

    # ── SL/TP ─────────────────────────────────────────────────────────────────
    sl_tp:            SLTPConfig      = field(default_factory=SLTPConfig)

    # ── Risk ──────────────────────────────────────────────────────────────────
    risk:             RiskConfig      = field(default_factory=RiskConfig)

    # ── Capital ───────────────────────────────────────────────────────────────
    initial_capital:  float = 10_000.0
    symbol:           str   = ""

    # ── Misc ──────────────────────────────────────────────────────────────────
    random_seed:      int   = 42
    skip_weekends:    bool  = True


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Complete backtest result."""
    config:              BacktestConfig
    trades:              list[Trade]
    metrics:             dict
    equity_df:           pd.DataFrame
    report_paths:        dict[str, Path]
    session_analysis:    dict
    direction_analysis:  dict
    confidence_bands:    list[dict]
    regime_analysis:     dict
    errors:              list[str] = field(default_factory=list)
    total_time_s:        float     = 0.0
    n_signals:           int       = 0
    n_filtered_signals:  int       = 0


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_backtest(strategy, data):
    """Legacy shim.  Use Backtester.run() for full institutional backtesting."""
    return None


class Backtester:
    """Institutional AI Backtesting Engine."""

    def __init__(self, config: BacktestConfig) -> None:
        self.cfg = config
        self._exec_engine = ExecutionEngine(config.execution, config.random_seed)
        self._sl_tp_mgr   = SLTPManager(config.sl_tp)
        self._pos_mgr     = PositionManager(config.position)
        self._risk_mgr    = RiskManager(RiskConfig(
            max_open_positions  = config.risk.max_open_positions,
            max_daily_loss_pct  = config.risk.max_daily_loss_pct,
            max_weekly_loss_pct = config.risk.max_weekly_loss_pct,
            min_confidence      = config.min_probability,
            initial_capital     = config.initial_capital,
        ))
        self._portfolio   = Portfolio(config.initial_capital)
        self._trade_engine = TradeEngine(min_probability=config.min_probability)

    def run(
        self,
        price_df:   Optional[pd.DataFrame] = None,
        bundle_dir: Optional[Path]         = None,
    ) -> BacktestResult:
        """Run the full backtest.

        Args:
            price_df:   OHLCV + feature DataFrame.  If None, loaded from config.
            bundle_dir: Override bundle_dir from config.

        Returns:
            BacktestResult with all metrics, trades, and report paths.
        """
        import time
        t0 = time.perf_counter()
        errors: list[str] = []

        # ── 1. Load price data ────────────────────────────────────────────────
        if price_df is None:
            if self.cfg.price_data_path is None:
                raise ValueError("price_df or config.price_data_path must be provided")
            price_df = pd.read_parquet(self.cfg.price_data_path)

        price_df = self._prepare_price_df(price_df)
        n_bars   = len(price_df)
        logger.info("Backtesting %d bars for %s", n_bars, self.cfg.symbol)

        # ── 2. Load inference pipeline ────────────────────────────────────────
        b_dir    = Path(bundle_dir or self.cfg.bundle_dir)
        pipeline = ArtifactManager.load_bundle(b_dir)
        logger.info("Loaded bundle: %s from %s", pipeline.model_name, b_dir)

        # ── 3. Generate predictions (read-only, run once) ─────────────────────
        predictions  = pipeline.predict(price_df)
        probabilities = pipeline.predict_proba(price_df)
        logger.info("Generated %d predictions", len(predictions))

        # ── 4. Convert to signals ─────────────────────────────────────────────
        ts_col = self.cfg.timestamp_column
        if ts_col in price_df.columns:
            timestamps = price_df[ts_col]
        else:
            timestamps = price_df.index.to_series()

        signals = self._trade_engine.generate_signals(
            timestamps    = timestamps,
            predictions   = predictions,
            probabilities = probabilities,
        )
        n_actionable = sum(1 for s in signals if s.is_actionable)
        logger.info("%d actionable signals (before risk filter)", n_actionable)

        # ── 5. Bar-by-bar simulation ──────────────────────────────────────────
        pending_entries: list[tuple[TradeSignal, Trade]] = []  # (signal, trade)

        for bar_idx, signal in enumerate(signals):
            row = price_df.iloc[bar_idx]
            bar_open  = float(row.get("open",  row.get("close", 0.0)))
            bar_high  = float(row.get("high",  bar_open))
            bar_low   = float(row.get("low",   bar_open))
            bar_close = float(row.get("close", bar_open))

            # ── a. Fill pending entries at this bar's open ────────────────────
            still_pending = []
            for sig, trade in pending_entries:
                if self._exec_engine.entry_bar(sig.bar_idx) == bar_idx:
                    ep, spread_cost, slip_cost = self._exec_engine.calculate_entry(
                        sig.direction, bar_open
                    )
                    sl, tp = self._sl_tp_mgr.compute_initial_levels(
                        sig.direction, ep, bar_idx, price_df
                    )
                    trade.entry_price    = ep
                    trade.entry_time     = signal.timestamp
                    trade.stop_loss      = sl
                    trade.take_profit    = tp
                    trade.spread_cost    = spread_cost
                    trade.slippage_cost  = slip_cost
                    trade.commission     = self._exec_engine.calculate_commission(trade.lot_size)
                    trade.entry_bar_idx  = bar_idx
                    trade.atr_at_entry   = float(row.get(self.cfg.sl_tp.atr_column, 0.0)) or None
                    trade.session        = get_session(signal.timestamp)
                    self._portfolio.open_trade(trade)
                else:
                    still_pending.append((sig, trade))
            pending_entries = still_pending

            # ── b. Update & check SL/TP for all open trades ───────────────────
            to_close: list[tuple[Trade, float, str]] = []   # (trade, price, reason)
            for trade in list(self._portfolio.open_trades):
                bars_held = bar_idx - (trade.entry_bar_idx or bar_idx)

                new_sl, new_tp, be_act, time_exit = self._sl_tp_mgr.update(
                    direction     = trade.direction,
                    entry_price   = trade.entry_price,
                    current_price = bar_close,
                    current_sl    = trade.stop_loss,
                    current_tp    = trade.take_profit,
                    be_activated  = trade.be_activated,
                    bars_held     = bars_held,
                )
                trade.stop_loss    = new_sl
                trade.take_profit  = new_tp
                trade.be_activated = be_act
                trade.trailing_sl  = new_sl if self.cfg.sl_tp.enable_trailing else None

                if time_exit:
                    exit_px = self._exec_engine.calculate_exit(trade.direction, bar_close)
                    to_close.append((trade, exit_px, "time_stop"))
                    continue

                hit = self._sl_tp_mgr.check_sl_tp_hit(
                    trade.direction, bar_high, bar_low,
                    trade.stop_loss, trade.take_profit,
                )
                if hit:
                    reason, hit_price = hit
                    exit_px = self._exec_engine.calculate_exit(
                        trade.direction, hit_price, is_limit_exit=True
                    )
                    to_close.append((trade, exit_px, reason))

            for trade, exit_px, reason in to_close:
                closed = self._portfolio.close_trade(
                    trade        = trade,
                    exit_price   = exit_px,
                    exit_reason  = reason,
                    exit_time    = signal.timestamp,
                    exit_bar_idx = bar_idx,
                    pip_size     = self.cfg.execution.pip_size,
                    pip_value    = self.cfg.execution.pip_value,
                )
                self._risk_mgr.record_closed_trade(signal.timestamp, closed.net_profit or 0.0)
                self._pos_mgr.record_trade_outcome(
                    closed.is_winner or False,
                    abs(closed.profit_pips or 0.0),
                )

            # ── c. Evaluate new signal ────────────────────────────────────────
            if signal.is_actionable:
                open_dirs = [t.direction for t in self._portfolio.open_trades]
                allowed, reason = self._risk_mgr.check(
                    timestamp        = signal.timestamp,
                    direction        = signal.direction,
                    confidence       = signal.confidence,
                    n_open_positions = len(self._portfolio.open_trades),
                    open_directions  = open_dirs,
                )
                if allowed:
                    # Compute SL for lot-sizing (use placeholder entry)
                    approx_entry = bar_close
                    approx_sl, approx_tp = self._sl_tp_mgr.compute_initial_levels(
                        signal.direction, approx_entry, bar_idx, price_df
                    )
                    lot = self._pos_mgr.compute_lot(
                        balance      = self._portfolio.balance,
                        direction    = signal.direction,
                        entry_price  = approx_entry,
                        stop_loss    = approx_sl,
                        bar_idx      = bar_idx,
                        price_df     = price_df,
                        confidence   = signal.confidence,
                    )
                    lot = self._exec_engine.clamp_lot(lot)

                    trade = Trade(
                        trade_id         = str(uuid.uuid4())[:8],
                        direction        = signal.direction,
                        signal_time      = signal.timestamp,
                        lot_size         = lot,
                        confidence       = signal.confidence,
                        prediction_class = signal.prediction_class,
                    )
                    entry_bar = self._exec_engine.entry_bar(bar_idx)
                    if entry_bar == bar_idx:
                        ep, spread_cost, slip_cost = self._exec_engine.calculate_entry(
                            signal.direction, bar_open
                        )
                        sl, tp = self._sl_tp_mgr.compute_initial_levels(
                            signal.direction, ep, bar_idx, price_df
                        )
                        trade.entry_price   = ep
                        trade.entry_time    = signal.timestamp
                        trade.stop_loss     = sl
                        trade.take_profit   = tp
                        trade.spread_cost   = spread_cost
                        trade.slippage_cost = slip_cost
                        trade.commission    = self._exec_engine.calculate_commission(lot)
                        trade.entry_bar_idx = bar_idx
                        trade.atr_at_entry  = float(row.get(self.cfg.sl_tp.atr_column, 0.0)) or None
                        trade.session       = get_session(signal.timestamp)
                        self._portfolio.open_trade(trade)
                    else:
                        pending_entries.append((signal, trade))

            # ── d. Record equity snapshot ────────────────────────────────────
            self._portfolio.record_equity(
                timestamp     = signal.timestamp,
                bar_idx       = bar_idx,
                current_price = bar_close,
                pip_size      = self.cfg.execution.pip_size,
                pip_value     = self.cfg.execution.pip_value,
            )

        # ── 6. Close remaining open trades at last bar ────────────────────────
        last_row    = price_df.iloc[-1]
        last_close  = float(last_row.get("close", last_row.get("open", 0.0)))
        last_ts     = signals[-1].timestamp if signals else pd.Timestamp.now()

        for trade in list(self._portfolio.open_trades):
            exit_px = self._exec_engine.calculate_exit(trade.direction, last_close)
            self._portfolio.close_trade(
                trade        = trade,
                exit_price   = exit_px,
                exit_reason  = "end_of_data",
                exit_time    = last_ts,
                exit_bar_idx = n_bars - 1,
                pip_size     = self.cfg.execution.pip_size,
                pip_value    = self.cfg.execution.pip_value,
            )

        # ── 7. Compute performance metrics ────────────────────────────────────
        closed  = self._portfolio.closed_trades
        profits = [t.net_profit for t in closed if t.net_profit is not None]
        eq_df   = self._portfolio.equity_dataframe()
        eq_series = eq_df["equity"] if not eq_df.empty and "equity" in eq_df.columns else pd.Series(dtype=float)

        metrics = compute_performance_metrics(
            net_profits  = profits,
            equity_curve = eq_series if not eq_series.empty else None,
            trades       = closed,
        )

        # ── 8. Analytics ──────────────────────────────────────────────────────
        session_analysis   = analyze_session_performance(closed)
        direction_analysis = analyze_direction_performance(closed)
        confidence_bands   = analyze_confidence_bands(closed)
        regime_analysis    = analyze_regime_performance(closed, price_df)

        # ── 9. Write reports ──────────────────────────────────────────────────
        output_dir = Path(self.cfg.output_dir)
        reporter   = BacktestReporter(output_dir)
        cfg_summary = {
            "symbol":           self.cfg.symbol,
            "bundle_dir":       str(b_dir),
            "initial_capital":  self.cfg.initial_capital,
            "min_probability":  self.cfg.min_probability,
            "sl_tp_mode":       self.cfg.sl_tp.mode,
            "position_mode":    self.cfg.position.mode,
            "n_bars":           n_bars,
        }
        report_paths = reporter.generate(
            trades         = closed,
            metrics        = metrics,
            equity_df      = eq_df,
            config_summary = cfg_summary,
        )

        # Save metrics JSON
        import json
        metrics_path = output_dir / "performance_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in metrics.items() if k != "exit_breakdown"}, f, indent=2)
        report_paths["performance_metrics"] = metrics_path

        total_time = time.perf_counter() - t0
        logger.info(
            "Backtest complete: %d trades, net=%.2f, time=%.1fs",
            len(closed), metrics.get("net_profit", 0.0), total_time,
        )

        return BacktestResult(
            config             = self.cfg,
            trades             = closed,
            metrics            = metrics,
            equity_df          = eq_df,
            report_paths       = report_paths,
            session_analysis   = session_analysis,
            direction_analysis = direction_analysis,
            confidence_bands   = confidence_bands,
            regime_analysis    = regime_analysis,
            errors             = errors,
            total_time_s       = round(total_time, 2),
            n_signals          = len(signals),
            n_filtered_signals = n_actionable,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prepare_price_df(self, df: pd.DataFrame) -> pd.DataFrame:
        ts_col = self.cfg.timestamp_column
        if ts_col in df.columns:
            df = df.sort_values(ts_col).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        return df
