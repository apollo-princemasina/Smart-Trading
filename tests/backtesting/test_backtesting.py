"""
Tests for the Institutional AI Backtesting Engine.

Design principles
-----------------
  - No ML training happens in any test.
  - Predictions are always synthetic arrays (read-only guarantee verified).
  - InferencePipeline is mocked to return fixed predictions without disk I/O.
  - Each test is independent; no shared mutable state.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_price_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    price  = 1.10000
    rows   = []
    ts     = pd.date_range("2024-01-01", periods=n, freq="H", tz="UTC")
    for i in range(n):
        price  += rng.normal(0, 0.0002)
        open_  = round(price, 5)
        high   = round(price + abs(rng.normal(0, 0.0001)), 5)
        low    = round(price - abs(rng.normal(0, 0.0001)), 5)
        close  = round(price + rng.normal(0, 0.0001), 5)
        atr    = 0.0010
        rows.append({
            "time":  ts[i],
            "open":  open_,
            "high":  high,
            "low":   low,
            "close": close,
            "atr":   atr,
        })
    df = pd.DataFrame(rows)
    return df


def _make_mock_pipeline(n: int, seed: int = 42) -> MagicMock:
    """Return a mock InferencePipeline that produces fixed predictions."""
    rng = np.random.default_rng(seed)
    predictions  = rng.integers(0, 2, size=n)
    proba_col1   = rng.uniform(0.6, 0.9, size=n)
    proba_col0   = 1.0 - proba_col1
    probabilities = np.column_stack([proba_col0, proba_col1])

    pipeline = MagicMock()
    pipeline.model_name   = "mock_model"
    pipeline.predict.return_value       = predictions
    pipeline.predict_proba.return_value = probabilities
    return pipeline


# ── Portfolio tests ───────────────────────────────────────────────────────────

class TestPortfolio:
    def test_initial_state(self):
        from src.backtesting.portfolio import Portfolio
        p = Portfolio(10_000.0)
        assert p.balance == 10_000.0
        assert p.initial_capital == 10_000.0
        assert p.open_trades == []
        assert p.closed_trades == []

    def test_open_trade_deducts_commission(self):
        from src.backtesting.portfolio import Portfolio, Trade
        p = Portfolio(10_000.0)
        t = Trade(trade_id="t1", direction="BUY", signal_time=pd.Timestamp.now(),
                  lot_size=1.0, commission=7.0)
        p.open_trade(t)
        assert p.balance == 10_000.0 - 7.0
        assert len(p.open_trades) == 1

    def test_close_trade_updates_balance(self):
        from src.backtesting.portfolio import Portfolio, Trade
        p = Portfolio(10_000.0)
        ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")
        t  = Trade(trade_id="t1", direction="BUY", signal_time=ts,
                   lot_size=1.0, commission=7.0, entry_price=1.10000,
                   entry_bar_idx=0)
        p.open_trade(t)
        p.close_trade(t, exit_price=1.10200, exit_reason="tp",
                      exit_time=ts, exit_bar_idx=5)
        assert len(p.closed_trades) == 1
        assert len(p.open_trades) == 0
        assert p.closed_trades[0].profit_pips == pytest.approx(20.0, abs=0.1)

    def test_equity_history_grows(self):
        from src.backtesting.portfolio import Portfolio
        p  = Portfolio(10_000.0)
        ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")
        p.record_equity(ts, 0, 1.10000)
        p.record_equity(ts, 1, 1.10100)
        assert len(p.equity_history) == 2

    def test_reject_trade_marks_status(self):
        from src.backtesting.portfolio import Portfolio, Trade
        p = Portfolio(10_000.0)
        t = Trade(trade_id="t1", direction="BUY", signal_time=pd.Timestamp.now())
        p.reject_trade(t, reason="max_positions")
        assert t.status == "rejected"
        assert t.exit_reason == "max_positions"

    def test_equity_dataframe_structure(self):
        from src.backtesting.portfolio import Portfolio
        p  = Portfolio(10_000.0)
        ts = pd.date_range("2024-01-01", periods=5, freq="H", tz="UTC")
        for i, t in enumerate(ts):
            p.record_equity(t, i, 1.10000)
        df = p.equity_dataframe()
        assert set(["balance", "equity", "drawdown", "open_trades"]).issubset(df.columns)
        assert len(df) == 5


# ── Trade dataclass tests ─────────────────────────────────────────────────────

class TestTrade:
    def test_is_winner_true(self):
        from src.backtesting.portfolio import Trade
        t = Trade(trade_id="x", direction="BUY", signal_time=pd.Timestamp.now(),
                  net_profit=50.0)
        assert t.is_winner is True

    def test_is_winner_false(self):
        from src.backtesting.portfolio import Trade
        t = Trade(trade_id="x", direction="SELL", signal_time=pd.Timestamp.now(),
                  net_profit=-30.0)
        assert t.is_winner is False

    def test_unrealized_pnl_buy(self):
        from src.backtesting.portfolio import Trade
        t = Trade(trade_id="x", direction="BUY", signal_time=pd.Timestamp.now(),
                  status="open", entry_price=1.10000, lot_size=1.0, commission=0.0)
        pnl = t.unrealized_pnl(1.10200, pip_size=0.0001, pip_value=10.0)
        assert pnl == pytest.approx(200.0, abs=1.0)   # 20 pips × 10 USD × 1 lot

    def test_unrealized_pnl_closed_returns_zero(self):
        from src.backtesting.portfolio import Trade
        t = Trade(trade_id="x", direction="BUY", signal_time=pd.Timestamp.now(),
                  status="closed", entry_price=1.10000, lot_size=1.0)
        assert t.unrealized_pnl(1.11000, pip_size=0.0001, pip_value=10.0) == 0.0


# ── TradeEngine tests ─────────────────────────────────────────────────────────

class TestTradeEngine:
    def _ts(self, n):
        return pd.date_range("2024-01-01", periods=n, freq="H", tz="UTC").to_series().reset_index(drop=True)

    def test_binary_buy_signal(self):
        from src.backtesting.trade_engine import TradeEngine
        engine = TradeEngine(min_probability=0.0)
        preds  = np.array([1])
        probas = np.array([[0.2, 0.8]])
        sigs   = engine.generate_signals(self._ts(1), preds, probas)
        assert sigs[0].direction == "BUY"
        assert sigs[0].confidence == pytest.approx(0.8)

    def test_binary_sell_signal(self):
        from src.backtesting.trade_engine import TradeEngine
        engine = TradeEngine(min_probability=0.0)
        preds  = np.array([0])
        probas = np.array([[0.85, 0.15]])
        sigs   = engine.generate_signals(self._ts(1), preds, probas)
        assert sigs[0].direction == "SELL"

    def test_ternary_hold(self):
        from src.backtesting.trade_engine import TradeEngine
        engine = TradeEngine(min_probability=0.0)
        preds  = np.array([1])
        probas = np.array([[0.1, 0.85, 0.05]])
        sigs   = engine.generate_signals(self._ts(1), preds, probas)
        assert sigs[0].direction == "HOLD"

    def test_confidence_filter(self):
        from src.backtesting.trade_engine import TradeEngine
        engine = TradeEngine(min_probability=0.75)
        preds  = np.array([1, 1])
        probas = np.array([[0.3, 0.70], [0.2, 0.80]])
        sigs   = engine.generate_signals(self._ts(2), preds, probas)
        assert sigs[0].direction == "HOLD"   # 0.70 < 0.75
        assert sigs[1].direction == "BUY"    # 0.80 >= 0.75

    def test_no_probabilities_falls_back(self):
        from src.backtesting.trade_engine import TradeEngine
        engine = TradeEngine(min_probability=0.0)
        preds  = np.array([1, 0])
        sigs   = engine.generate_signals(self._ts(2), preds, None)
        assert sigs[0].direction == "BUY"
        assert sigs[1].direction == "SELL"
        assert sigs[0].confidence == 0.5

    def test_custom_class_map(self):
        from src.backtesting.trade_engine import TradeEngine
        engine = TradeEngine(
            min_probability=0.0,
            prediction_class_map={0: "SELL", 1: "BUY", 2: "HOLD"},
        )
        preds = np.array([2])
        sigs  = engine.generate_signals(self._ts(1), preds, None)
        assert sigs[0].direction == "HOLD"


# ── ExecutionEngine tests ────────────────────────────────────────────────────

class TestExecutionEngine:
    def _make_engine(self, **kwargs):
        from src.backtesting.execution_engine import ExecutionConfig, ExecutionEngine
        cfg = ExecutionConfig(**kwargs)
        return ExecutionEngine(cfg, random_seed=0)

    def test_entry_bar_delayed(self):
        eng = self._make_engine(execution_delay_bars=2)
        assert eng.entry_bar(5) == 7

    def test_buy_entry_price_above_open(self):
        eng = self._make_engine(spread_pips=2.0, slippage_pips=0.0, slippage_std=0.0)
        ep, spread_cost, slip_cost = eng.calculate_entry("BUY", 1.10000)
        assert ep > 1.10000
        assert spread_cost > 0.0

    def test_sell_entry_price_below_open(self):
        eng = self._make_engine(spread_pips=2.0, slippage_pips=0.0, slippage_std=0.0)
        ep, _, _ = eng.calculate_entry("SELL", 1.10000)
        assert ep < 1.10000

    def test_limit_exit_no_slippage(self):
        eng = self._make_engine()
        exit_px = eng.calculate_exit("BUY", 1.10200, is_limit_exit=True)
        assert exit_px == pytest.approx(1.10200)

    def test_commission_proportional_to_lot(self):
        eng = self._make_engine(commission_per_lot=7.0)
        assert eng.calculate_commission(2.0) == pytest.approx(14.0)

    def test_clamp_lot(self):
        eng = self._make_engine()
        assert eng.clamp_lot(0.001) == pytest.approx(0.01)
        assert eng.clamp_lot(200.0) == pytest.approx(100.0)


# ── SLTPManager tests ────────────────────────────────────────────────────────

class TestSLTPManager:
    def _price_df(self):
        return pd.DataFrame({
            "open":  [1.10000] * 10,
            "high":  [1.10500] * 10,
            "low":   [1.09500] * 10,
            "close": [1.10000] * 10,
            "atr":   [0.0010]  * 10,
        })

    def test_fixed_pips_buy(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig(mode="fixed_pips", sl_pips=20.0, tp_pips=40.0))
        sl, tp = mgr.compute_initial_levels("BUY", 1.10000, 5, self._price_df())
        assert sl == pytest.approx(1.09800, abs=1e-5)
        assert tp == pytest.approx(1.10400, abs=1e-5)

    def test_fixed_pips_sell(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig(mode="fixed_pips", sl_pips=20.0, tp_pips=40.0))
        sl, tp = mgr.compute_initial_levels("SELL", 1.10000, 5, self._price_df())
        assert sl == pytest.approx(1.10200, abs=1e-5)
        assert tp == pytest.approx(1.09600, abs=1e-5)

    def test_atr_sl_tp(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig(mode="atr", sl_atr_mult=1.5, tp_atr_mult=3.0))
        sl, tp = mgr.compute_initial_levels("BUY", 1.10000, 5, self._price_df())
        expected_sl = 1.10000 - 1.5 * 0.0010
        expected_tp = 1.10000 + 3.0 * 0.0010
        assert sl == pytest.approx(expected_sl, abs=1e-5)
        assert tp == pytest.approx(expected_tp, abs=1e-5)

    def test_sl_hit_long(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig())
        result = mgr.check_sl_tp_hit("BUY", bar_high=1.10100, bar_low=1.09700,
                                     stop_loss=1.09800, take_profit=1.10400)
        assert result == ("sl", 1.09800)

    def test_tp_hit_long(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig())
        result = mgr.check_sl_tp_hit("BUY", bar_high=1.10500, bar_low=1.10100,
                                     stop_loss=1.09800, take_profit=1.10400)
        assert result == ("tp", 1.10400)

    def test_no_hit(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig())
        result = mgr.check_sl_tp_hit("BUY", bar_high=1.10200, bar_low=1.10000,
                                     stop_loss=1.09800, take_profit=1.10400)
        assert result is None

    def test_time_stop_triggers(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig(enable_time_stop=True, max_holding_bars=5))
        _, _, _, reason = mgr.update("BUY", 1.10000, 1.10100, 1.09800, 1.10400,
                                      False, bars_held=5)
        assert reason == "time_stop"

    def test_trailing_sl_updates(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig(enable_trailing=True, trailing_pips=20.0))
        initial_sl = 1.09800
        new_sl, _, _, _ = mgr.update("BUY", 1.10000, 1.10300, initial_sl, 1.10400,
                                      False, bars_held=2)
        assert new_sl > initial_sl

    def test_break_even_activates(self):
        from src.backtesting.sl_tp_manager import SLTPConfig, SLTPManager
        mgr = SLTPManager(SLTPConfig(enable_break_even=True, be_trigger_rr=1.0,
                                      be_buffer_pips=2.0, sl_pips=20.0, tp_pips=40.0))
        # Price has moved to TP level → RR = 1.0 → BE should activate
        _, _, be_act, _ = mgr.update("BUY", 1.10000, 1.10400, 1.09800, 1.10400,
                                      False, bars_held=3)
        assert be_act is True


# ── PositionManager tests ────────────────────────────────────────────────────

class TestPositionManager:
    def _price_df(self):
        return pd.DataFrame({"atr": [0.0010] * 10})

    def test_fixed_lot(self):
        from src.backtesting.position_manager import PositionConfig, PositionManager
        mgr = PositionManager(PositionConfig(mode="fixed_lot", fixed_lot_size=0.5))
        lot = mgr.compute_lot(10000.0, "BUY", 1.1, 1.08, 0, self._price_df())
        assert lot == pytest.approx(0.5)

    def test_fixed_risk_pct(self):
        from src.backtesting.position_manager import PositionConfig, PositionManager
        mgr = PositionManager(PositionConfig(mode="fixed_risk_pct", risk_pct=0.01))
        lot = mgr.compute_lot(10000.0, "BUY", 1.10000, 1.09800, 0, self._price_df())
        expected_risk_usd = 100.0   # 1% of 10k
        sl_pips           = 20.0
        expected_lot      = expected_risk_usd / (sl_pips * 10.0)
        assert lot == pytest.approx(expected_lot, rel=0.05)

    def test_clamp_min(self):
        from src.backtesting.position_manager import PositionConfig, PositionManager
        mgr = PositionManager(PositionConfig(mode="fixed_lot", fixed_lot_size=0.001))
        lot = mgr.compute_lot(100.0, "BUY", 1.1, 1.09, 0, self._price_df())
        assert lot >= mgr.cfg.min_lot

    def test_clamp_max(self):
        from src.backtesting.position_manager import PositionConfig, PositionManager
        mgr = PositionManager(PositionConfig(mode="fixed_lot", fixed_lot_size=1000.0, max_lot=10.0))
        lot = mgr.compute_lot(10000.0, "BUY", 1.1, 1.09, 0, self._price_df())
        assert lot <= 10.0


# ── RiskManager tests ────────────────────────────────────────────────────────

class TestRiskManager:
    def _ts(self, day: str = "2024-01-02"):
        return pd.Timestamp(f"{day} 10:00", tz="UTC")

    def test_allows_valid_signal(self):
        from src.backtesting.risk_manager import RiskConfig, RiskManager
        mgr = RiskManager(RiskConfig(max_open_positions=3, min_confidence=0.6))
        ok, reason = mgr.check(self._ts(), "BUY", 0.70, 0, [])
        assert ok is True
        assert reason == "ok"

    def test_blocks_max_positions(self):
        from src.backtesting.risk_manager import RiskConfig, RiskManager
        mgr = RiskManager(RiskConfig(max_open_positions=2))
        ok, reason = mgr.check(self._ts(), "BUY", 0.80, 2, [])
        assert ok is False
        assert reason == "max_positions"

    def test_blocks_low_confidence(self):
        from src.backtesting.risk_manager import RiskConfig, RiskManager
        mgr = RiskManager(RiskConfig(min_confidence=0.70))
        ok, reason = mgr.check(self._ts(), "BUY", 0.65, 0, [])
        assert ok is False
        assert reason == "low_confidence"

    def test_blocks_opposing_direction(self):
        from src.backtesting.risk_manager import RiskConfig, RiskManager
        mgr = RiskManager(RiskConfig(allow_simultaneous_rr=False))
        ok, reason = mgr.check(self._ts(), "BUY", 0.80, 0, ["SELL"])
        assert ok is False
        assert reason == "opposing_position"

    def test_allows_same_direction(self):
        from src.backtesting.risk_manager import RiskConfig, RiskManager
        mgr = RiskManager(RiskConfig(max_open_positions=3, allow_simultaneous_rr=False))
        ok, _ = mgr.check(self._ts(), "BUY", 0.80, 1, ["BUY"])
        assert ok is True

    def test_daily_loss_limit(self):
        from src.backtesting.risk_manager import RiskConfig, RiskManager
        mgr = RiskManager(RiskConfig(max_daily_loss_pct=0.02, initial_capital=10_000.0))
        mgr.record_closed_trade(self._ts(), -210.0)   # > 2% of 10k
        ok, reason = mgr.check(self._ts(), "BUY", 0.80, 0, [])
        assert ok is False
        assert reason == "daily_loss_limit"

    def test_weekly_loss_limit(self):
        from src.backtesting.risk_manager import RiskConfig, RiskManager
        # Use a very low weekly limit (0.5%) and spread losses across different days
        # of the same week so the daily limit (2%) is never hit individually.
        mgr = RiskManager(RiskConfig(
            max_daily_loss_pct=0.10,   # 10% daily — won't trigger
            max_weekly_loss_pct=0.01,  # 1% weekly — will trigger at $100
            initial_capital=10_000.0,
        ))
        # Record $60 losses on Mon/Tue/Wed of the same ISO week
        for day in ["2024-01-08", "2024-01-09", "2024-01-10"]:  # Mon–Wed W02
            mgr.record_closed_trade(pd.Timestamp(f"{day} 10:00", tz="UTC"), -60.0)
        ts = pd.Timestamp("2024-01-11 10:00", tz="UTC")  # same week
        ok, reason = mgr.check(ts, "BUY", 0.80, 0, [])
        assert ok is False
        assert reason == "weekly_loss_limit"


# ── Performance metrics tests ────────────────────────────────────────────────

class TestPerformanceMetrics:
    def test_empty_trades(self):
        from src.backtesting.performance import compute_performance_metrics
        m = compute_performance_metrics([], None)
        assert isinstance(m, dict)

    def test_basic_counts(self):
        from src.backtesting.performance import compute_performance_metrics
        profits = [100.0, -50.0, 75.0, -25.0, 60.0]
        m = compute_performance_metrics(profits, None)
        assert m["n_trades"]  == 5
        assert m["n_winners"] == 3
        assert m["n_losers"]  == 2
        assert m["win_rate"]  == pytest.approx(0.6)

    def test_profit_factor(self):
        from src.backtesting.performance import compute_performance_metrics
        profits = [200.0, -100.0]   # PF = 2.0
        m = compute_performance_metrics(profits, None)
        assert m["profit_factor"] == pytest.approx(2.0)

    def test_drawdown_metrics(self):
        from src.backtesting.performance import compute_drawdown_metrics
        eq = pd.Series([10000, 10200, 10100, 10300, 10050, 10400])
        m  = compute_drawdown_metrics(eq)
        assert m["max_drawdown"] <= 0
        assert m["max_drawdown_pct"] <= 0

    def test_sharpe_non_none_with_equity(self):
        from src.backtesting.performance import compute_performance_metrics
        eq = pd.Series(
            [10000 + i * 10 for i in range(50)],
            index=pd.date_range("2024-01-01", periods=50, freq="H", tz="UTC"),
        )
        m  = compute_performance_metrics([10.0] * 50, eq)
        assert m["sharpe_ratio"] is not None

    def test_omega_ratio(self):
        from src.backtesting.performance import compute_return_metrics
        # Mix of positive and negative returns so omega_ratio is computable
        rng = np.random.default_rng(0)
        vals = 10000 + np.cumsum(rng.normal(0, 50, 50))
        eq = pd.Series(
            vals,
            index=pd.date_range("2024-01-01", periods=50, freq="H", tz="UTC"),
        )
        m  = compute_return_metrics(eq)
        assert m["omega_ratio"] is not None

    def test_ulcer_index(self):
        from src.backtesting.performance import compute_drawdown_metrics
        eq = pd.Series([10000, 9800, 9600, 9900, 10100])
        m  = compute_drawdown_metrics(eq)
        assert m["ulcer_index"] is not None
        assert m["ulcer_index"] >= 0

    def test_consecutive_wins(self):
        from src.backtesting.performance import compute_performance_metrics
        profits = [10, 10, 10, -5, 10, 10]
        m = compute_performance_metrics(profits, None)
        assert m["max_consecutive_wins"] == 3

    def test_period_returns_monthly(self):
        from src.backtesting.performance import compute_period_returns
        eq = pd.Series(
            [10000 + i * 10 for i in range(60)],
            index=pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
        )
        df = compute_period_returns(eq, period="ME")
        assert isinstance(df, pd.DataFrame)
        assert "return_pct" in df.columns


# ── Analytics tests ───────────────────────────────────────────────────────────

class TestAnalytics:
    def test_london_session(self):
        from src.backtesting.analytics import get_session
        ts = pd.Timestamp("2024-01-15 09:30", tz="UTC")
        assert get_session(ts) == "london"

    def test_newyork_session(self):
        from src.backtesting.analytics import get_session
        ts = pd.Timestamp("2024-01-15 18:00", tz="UTC")
        assert get_session(ts) == "newyork"

    def test_overlap_session(self):
        from src.backtesting.analytics import get_session
        ts = pd.Timestamp("2024-01-15 14:00", tz="UTC")
        assert get_session(ts) == "overlap"

    def test_asian_session(self):
        from src.backtesting.analytics import get_session
        ts = pd.Timestamp("2024-01-15 03:00", tz="UTC")
        assert get_session(ts) == "asian"

    def test_offhours(self):
        from src.backtesting.analytics import get_session
        ts = pd.Timestamp("2024-01-15 23:00", tz="UTC")
        assert get_session(ts) == "offhours"

    def test_classify_regime_returns_string(self):
        from src.backtesting.analytics import classify_regime
        df = pd.DataFrame({
            "close": [1.10 + i * 0.0001 for i in range(30)],
            "high":  [1.10 + i * 0.0001 + 0.0005 for i in range(30)],
            "low":   [1.10 + i * 0.0001 - 0.0005 for i in range(30)],
            "atr":   [0.0010] * 30,
        })
        regime = classify_regime(df, 25)
        assert regime in ("trending", "ranging", "high_vol", "low_vol", "unknown")

    def test_confidence_bands_coverage(self):
        from src.backtesting.analytics import analyze_confidence_bands
        from src.backtesting.portfolio import Trade
        trades = [
            Trade(trade_id=str(i), direction="BUY", signal_time=pd.Timestamp.now(),
                  status="closed", net_profit=10.0 if i % 2 == 0 else -5.0,
                  confidence=0.60 + i * 0.04)
            for i in range(10)
        ]
        bands = analyze_confidence_bands(trades)
        assert len(bands) > 0
        assert all("win_rate" in b for b in bands)

    def test_direction_performance(self):
        from src.backtesting.analytics import analyze_direction_performance
        from src.backtesting.portfolio import Trade
        trades = [
            Trade(trade_id="b1", direction="BUY",  signal_time=pd.Timestamp.now(),
                  status="closed", net_profit=50.0),
            Trade(trade_id="s1", direction="SELL", signal_time=pd.Timestamp.now(),
                  status="closed", net_profit=-20.0),
        ]
        result = analyze_direction_performance(trades)
        assert "BUY"  in result
        assert "SELL" in result
        assert result["BUY"]["win_rate"] == 1.0

    def test_session_performance(self):
        from src.backtesting.analytics import analyze_session_performance
        from src.backtesting.portfolio import Trade
        trades = [
            Trade(trade_id="l1", direction="BUY", signal_time=pd.Timestamp.now(),
                  status="closed", net_profit=30.0, session="london"),
            Trade(trade_id="l2", direction="BUY", signal_time=pd.Timestamp.now(),
                  status="closed", net_profit=-10.0, session="london"),
        ]
        result = analyze_session_performance(trades)
        assert "london" in result
        assert result["london"]["n_trades"] == 2
        assert result["london"]["win_rate"] == pytest.approx(0.5)


# ── BacktestReporter tests ────────────────────────────────────────────────────

class TestBacktestReporter:
    def test_generate_creates_all_files(self, tmp_path):
        from src.backtesting.portfolio import Trade
        from src.backtesting.reports import BacktestReporter

        trades = [
            Trade(
                trade_id="t1", direction="BUY",
                signal_time=pd.Timestamp("2024-01-15 10:00", tz="UTC"),
                entry_time=pd.Timestamp("2024-01-15 11:00", tz="UTC"),
                exit_time=pd.Timestamp("2024-01-16 10:00", tz="UTC"),
                entry_price=1.10000, exit_price=1.10400,
                lot_size=1.0, commission=7.0, spread_cost=1.0, slippage_cost=0.5,
                gross_profit=40.0, net_profit=31.5, profit_pips=40.0,
                status="closed", exit_reason="tp", holding_bars=24,
                confidence=0.75, prediction_class=1,
            )
        ]
        eq_idx = pd.date_range("2024-01-15", periods=48, freq="H", tz="UTC")
        equity_df = pd.DataFrame(
            {"equity": [10000 + i for i in range(48)],
             "balance": [10000 + i for i in range(48)]},
            index=eq_idx,
        )
        metrics = {
            "net_profit": 31.5, "gross_profit": 40.0, "gross_loss": 0.0,
            "profit_factor": None, "expectancy": 31.5,
            "win_rate": 1.0, "n_trades": 1, "n_winners": 1, "n_losers": 0,
            "sharpe_ratio": None, "sortino_ratio": None,
            "calmar_ratio": None, "omega_ratio": None,
            "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "recovery_factor": None, "ulcer_index": 0.0,
            "exit_breakdown": {"tp": 1},
        }
        reporter = BacktestReporter(tmp_path)
        paths    = reporter.generate(trades, metrics, equity_df, {"symbol": "EURUSD"})

        expected_files = [
            "backtest_report.md", "trade_log.csv", "equity_curve.csv",
            "performance_summary.csv", "risk_report.md", "trade_statistics.csv",
            "monthly_returns.csv", "yearly_returns.csv",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_trade_log_columns(self, tmp_path):
        from src.backtesting.portfolio import Trade
        from src.backtesting.reports import BacktestReporter
        trades = [
            Trade(trade_id="x", direction="BUY",
                  signal_time=pd.Timestamp.now(),
                  entry_time=pd.Timestamp.now(),
                  exit_time=pd.Timestamp.now(),
                  status="closed", net_profit=10.0)
        ]
        r = BacktestReporter(tmp_path)
        r._write_trade_log(trades)
        df = pd.read_csv(tmp_path / "trade_log.csv")
        assert "trade_id" in df.columns
        assert "direction" in df.columns
        assert "net_profit" in df.columns


# ── End-to-end integration test ───────────────────────────────────────────────

class TestBacktesterIntegration:
    """End-to-end test with a mocked InferencePipeline.

    Verifies that:
    1. No model.fit() is called (read-only contract).
    2. Predictions are consumed exactly as produced.
    3. The result contains all expected keys.
    """

    def _make_config(self, tmp_path: Path):
        from src.backtesting.backtester import BacktestConfig
        from src.backtesting.execution_engine import ExecutionConfig
        from src.backtesting.sl_tp_manager import SLTPConfig
        from src.backtesting.position_manager import PositionConfig
        from src.backtesting.risk_manager import RiskConfig

        return BacktestConfig(
            bundle_dir      = tmp_path / "bundle",
            output_dir      = tmp_path / "output",
            initial_capital = 10_000.0,
            min_probability = 0.60,
            execution       = ExecutionConfig(
                spread_pips=1.0, commission_per_lot=7.0,
                slippage_pips=0.0, slippage_std=0.0,
                execution_delay_bars=1,
            ),
            sl_tp           = SLTPConfig(
                mode="fixed_pips", sl_pips=20.0, tp_pips=40.0
            ),
            position        = PositionConfig(mode="fixed_lot", fixed_lot_size=0.10),
            risk            = RiskConfig(max_open_positions=3),
            symbol          = "EURUSD",
        )

    def test_no_model_fit_called(self, tmp_path):
        from src.backtesting.backtester import Backtester

        cfg      = self._make_config(tmp_path)
        price_df = _make_price_df(50)
        pipeline = _make_mock_pipeline(50)

        with patch("src.backtesting.backtester.ArtifactManager") as mock_am:
            mock_am.load_bundle.return_value = pipeline
            backtester = Backtester(cfg)
            result     = backtester.run(price_df=price_df)

        # fit() should never have been called
        pipeline.fit.assert_not_called()

    def test_predictions_not_modified(self, tmp_path):
        from src.backtesting.backtester import Backtester

        cfg      = self._make_config(tmp_path)
        n_bars   = 50
        price_df = _make_price_df(n_bars)
        pipeline = _make_mock_pipeline(n_bars)

        original_predictions = pipeline.predict.return_value.copy()

        with patch("src.backtesting.backtester.ArtifactManager") as mock_am:
            mock_am.load_bundle.return_value = pipeline
            Backtester(cfg).run(price_df=price_df)

        # Verify predictions were not modified
        np.testing.assert_array_equal(
            pipeline.predict.return_value, original_predictions
        )

    def test_result_structure(self, tmp_path):
        from src.backtesting.backtester import Backtester, BacktestResult

        cfg      = self._make_config(tmp_path)
        price_df = _make_price_df(50)
        pipeline = _make_mock_pipeline(50)

        with patch("src.backtesting.backtester.ArtifactManager") as mock_am:
            mock_am.load_bundle.return_value = pipeline
            result = Backtester(cfg).run(price_df=price_df)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, list)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.equity_df, pd.DataFrame)
        assert isinstance(result.report_paths, dict)
        assert result.n_signals == 50
        assert result.total_time_s >= 0.0

    def test_report_files_created(self, tmp_path):
        from src.backtesting.backtester import Backtester

        cfg      = self._make_config(tmp_path)
        price_df = _make_price_df(50)
        pipeline = _make_mock_pipeline(50)

        with patch("src.backtesting.backtester.ArtifactManager") as mock_am:
            mock_am.load_bundle.return_value = pipeline
            result = Backtester(cfg).run(price_df=price_df)

        output_dir = tmp_path / "output"
        assert (output_dir / "backtest_report.md").exists()
        assert (output_dir / "trade_log.csv").exists()
        assert (output_dir / "equity_curve.csv").exists()
        assert (output_dir / "performance_metrics.json").exists()

    def test_metrics_keys_present(self, tmp_path):
        from src.backtesting.backtester import Backtester

        cfg      = self._make_config(tmp_path)
        price_df = _make_price_df(50)
        pipeline = _make_mock_pipeline(50)

        with patch("src.backtesting.backtester.ArtifactManager") as mock_am:
            mock_am.load_bundle.return_value = pipeline
            result = Backtester(cfg).run(price_df=price_df)

        for key in ["n_trades", "win_rate", "net_profit", "profit_factor",
                    "max_drawdown", "sharpe_ratio"]:
            assert key in result.metrics

    def test_all_signals_consumed(self, tmp_path):
        """Every bar produces exactly one signal."""
        from src.backtesting.backtester import Backtester

        n_bars   = 30
        cfg      = self._make_config(tmp_path)
        price_df = _make_price_df(n_bars)
        pipeline = _make_mock_pipeline(n_bars)

        with patch("src.backtesting.backtester.ArtifactManager") as mock_am:
            mock_am.load_bundle.return_value = pipeline
            result = Backtester(cfg).run(price_df=price_df)

        assert result.n_signals == n_bars

    def test_equity_curve_length_matches_bars(self, tmp_path):
        from src.backtesting.backtester import Backtester

        n_bars   = 40
        cfg      = self._make_config(tmp_path)
        price_df = _make_price_df(n_bars)
        pipeline = _make_mock_pipeline(n_bars)

        with patch("src.backtesting.backtester.ArtifactManager") as mock_am:
            mock_am.load_bundle.return_value = pipeline
            result = Backtester(cfg).run(price_df=price_df)

        assert len(result.equity_df) == n_bars

    def test_balance_is_finite_after_run(self, tmp_path):
        from src.backtesting.backtester import Backtester

        cfg      = self._make_config(tmp_path)
        price_df = _make_price_df(60)
        pipeline = _make_mock_pipeline(60)

        with patch("src.backtesting.backtester.ArtifactManager") as mock_am:
            mock_am.load_bundle.return_value = pipeline
            result = Backtester(cfg).run(price_df=price_df)

        final_balance = result.equity_df["balance"].iloc[-1] if not result.equity_df.empty else result.metrics.get("net_profit", 0.0)
        assert math.isfinite(float(final_balance))
