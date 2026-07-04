"""Comprehensive tests for the Traditional Technical Indicator Engine.

Coverage:
    TestMAContract             — MovingAveragesEngine interface contract (6)
    TestMAOutputStructure      — column presence, dtype, shape (5)
    TestEMAValues              — EMA convergence and ordering (8)
    TestWMAHMA                 — WMA and HMA correctness (5)
    TestEMADerived             — ema_slope and ema_cross semantics (6)
    TestMomentumContract       — MomentumEngine interface contract (6)
    TestRSI                    — RSI bounds and known values (6)
    TestStochastic             — %K/%D bounds and smoothing (4)
    TestMACD                   — MACD/Signal/Histogram (5)
    TestCCIWilliamsROCTSI      — CCI, Williams %R, ROC, TSI bounds (8)
    TestTrendContract          — TrendEngine interface contract (6)
    TestADXDI                  — ADX/±DI bounds and direction (7)
    TestAroon                  — Aroon Up/Down/Oscillator (5)
    TestParabolicSAR           — PSAR flip detection and bounds (5)
    TestVolatilityContract     — VolatilityEngine interface contract (6)
    TestATR                    — ATR positivity and normalization (5)
    TestBollingerBands         — BB width/position/percent_b (6)
    TestKeltnerDonchian        — KC/DC containment (5)
    TestChaikinVolatility      — Chaikin Vol sign and ROC (3)
    TestOscillatorContract     — OscillatorsEngine interface contract (6)
    TestVWAP                   — VWAP daily reset and bounds (5)
    TestOBVAD                  — OBV/AD direction (4)
    TestCMFMFI                 — CMF/MFI bounds (5)
    TestForceIndexEOM          — Force Index and EOM sign (4)
    TestTechnicalEngineContract— TechnicalEngine interface contract (6)
    TestCrossIndicator         — price_vs_ema200, MACD norm, trend_strength (7)
    TestEdgeCases              — single row, zero volume, constant price (5)
    TestIntegrationAndDtype    — end-to-end float64, shape, registry (5)
    TestPerformance            — 10k bar timing (1)
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from src.features.feature_registry import FeatureRegistry
import src.features  # noqa: F401 — triggers all @register decorators

from src.features.technical.moving_averages  import MovingAveragesEngine, _wma
from src.features.technical.momentum        import MomentumEngine
from src.features.technical.trend           import TrendEngine, _parabolic_sar
from src.features.technical.volatility      import VolatilityEngine
from src.features.technical.oscillators     import OscillatorsEngine
from src.features.technical.technical_engine import TechnicalEngine

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Random-walk OHLCV with DatetimeIndex spanning multiple UTC days."""
    rng = np.random.default_rng(seed)
    mid = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))
    spread = 0.0002
    high   = mid + rng.uniform(0, spread, n)
    low    = mid - rng.uniform(0, spread, n)
    close  = mid + rng.normal(0, spread / 4, n)
    open_  = mid + rng.normal(0, spread / 4, n)
    # keep OHLC consistent
    high   = np.maximum(high, np.maximum(open_, close))
    low    = np.minimum(low,  np.minimum(open_, close))
    volume = rng.uniform(100, 10_000, n)
    idx    = pd.date_range("2024-01-02 00:00", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_trend_df(n: int = 300) -> pd.DataFrame:
    """Monotonically rising price for directional tests."""
    close  = np.linspace(1.0, 1.3, n)
    high   = close + 0.001
    low    = close - 0.001
    open_  = close - 0.0005
    volume = np.ones(n) * 1000
    idx    = pd.date_range("2024-01-02", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def rand_df() -> pd.DataFrame:
    return _make_ohlcv()


@pytest.fixture
def trend_df() -> pd.DataFrame:
    return _make_trend_df()


# ─────────────────────────────────────────────────────────────────────────────
# TestMAContract
# ─────────────────────────────────────────────────────────────────────────────

class TestMAContract:
    def test_registered(self):
        assert "moving_averages" in FeatureRegistry.all_features()

    def test_name(self):
        assert MovingAveragesEngine.name == "moving_averages"

    def test_category(self):
        assert MovingAveragesEngine.category == "technical"

    def test_dependencies_empty(self):
        assert MovingAveragesEngine.dependencies == []

    def test_required_columns(self):
        assert "close" in MovingAveragesEngine.required_columns

    def test_metadata_output_columns(self):
        eng = MovingAveragesEngine()
        meta = eng.metadata()
        assert len(meta.output_columns) == 12


# ─────────────────────────────────────────────────────────────────────────────
# TestMAOutputStructure
# ─────────────────────────────────────────────────────────────────────────────

class TestMAOutputStructure:
    def test_shape(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert out.shape == (len(rand_df), 12)

    def test_all_columns_present(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        expected = [
            "ema9", "ema20", "ema50", "ema100", "ema200",
            "sma20", "sma50", "sma100", "wma20", "hma20",
            "ema_slope", "ema_cross",
        ]
        assert list(out.columns) == expected

    def test_all_float64(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert (out.dtypes == np.float64).all()

    def test_no_nan(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert not out.isnull().any().any()

    def test_no_inf(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert not np.isinf(out.values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestEMAValues
# ─────────────────────────────────────────────────────────────────────────────

class TestEMAValues:
    def test_ema_responds_to_close(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        corr = np.corrcoef(rand_df["close"].values, out["ema20"].values)[0, 1]
        assert corr > 0.90

    def test_faster_ema_reacts_more(self, rand_df):
        """EMA9 should have higher variance than EMA200 (more reactive)."""
        out = MovingAveragesEngine().generate(rand_df)
        assert out["ema9"].std() > out["ema200"].std()

    def test_trend_ema_ordering(self, trend_df):
        """In a rising market EMA9 > EMA20 > EMA50 should hold at the tail."""
        out = MovingAveragesEngine().generate(trend_df)
        tail = out.tail(50)
        assert (tail["ema9"] > tail["ema20"]).all()
        assert (tail["ema20"] > tail["ema50"]).all()

    def test_sma_is_flat_on_constant(self):
        n = 100
        df = pd.DataFrame({"close": np.ones(n)},
                          index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = MovingAveragesEngine().generate(df)
        assert np.allclose(out["sma20"], 1.0)
        assert np.allclose(out["ema50"], 1.0, atol=1e-10)

    def test_sma20_vs_pandas(self, rand_df):
        out  = MovingAveragesEngine().generate(rand_df)
        expected = rand_df["close"].rolling(20, min_periods=1).mean().values
        np.testing.assert_allclose(out["sma20"].values, expected, rtol=1e-10)

    def test_ema9_matches_ewm(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        expected = rand_df["close"].ewm(span=9, adjust=False).mean().values
        np.testing.assert_allclose(out["ema9"].values, expected, rtol=1e-10)

    def test_ema200_converges_slowly(self, rand_df):
        """EMA200 should not fully converge in first 50 bars."""
        out = MovingAveragesEngine().generate(rand_df)
        # ema200 at bar 50 should be pulled strongly towards the first bar
        assert abs(out["ema200"].iloc[50] - rand_df["close"].iloc[0]) < \
               abs(out["ema9"].iloc[50]  - rand_df["close"].iloc[0])

    def test_wma_weights_recent_more(self):
        """WMA weights the most recent bar most — spike at end lifts WMA above SMA."""
        # 24 bars at 1.0 then one bar at 2.0 (window fully filled)
        vals = np.concatenate([np.ones(24), [2.0]])
        df   = pd.DataFrame({"close": vals},
                            index=pd.date_range("2024-01-01", periods=25, freq="h", tz="UTC"))
        out  = MovingAveragesEngine().generate(df)
        # WMA gives the 2.0 bar weight 20/210 while SMA weights it 1/20 equally
        assert out["wma20"].iloc[-1] > out["sma20"].iloc[-1]


# ─────────────────────────────────────────────────────────────────────────────
# TestWMAHMA
# ─────────────────────────────────────────────────────────────────────────────

class TestWMAHMA:
    def test_wma_known_value(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _wma(arr, 3)
        # weights [1,2,3]/6 → (1+4+9)/6 = 14/6 ≈ 2.333
        assert abs(result[-1] - 14 / 6) < 1e-10

    def test_wma_same_length_as_input(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert len(out["wma20"]) == len(rand_df)

    def test_hma_same_length(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert len(out["hma20"]) == len(rand_df)

    def test_hma_no_nan(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert not np.isnan(out["hma20"].values).any()

    def test_hma_more_responsive_than_sma20(self, trend_df):
        """HMA should track a trend more closely than SMA20."""
        out = MovingAveragesEngine().generate(trend_df)
        hma_err = np.abs(out["hma20"].values - trend_df["close"].values).mean()
        sma_err = np.abs(out["sma20"].values  - trend_df["close"].values).mean()
        assert hma_err < sma_err


# ─────────────────────────────────────────────────────────────────────────────
# TestEMADerived
# ─────────────────────────────────────────────────────────────────────────────

class TestEMADerived:
    def test_ema_cross_values_in_set(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert set(out["ema_cross"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_ema_cross_positive_in_uptrend(self, trend_df):
        out = MovingAveragesEngine().generate(trend_df)
        assert out["ema_cross"].tail(100).mean() > 0

    def test_ema_slope_zero_on_constant(self):
        n  = 100
        df = pd.DataFrame({"close": np.ones(n)},
                          index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = MovingAveragesEngine().generate(df)
        assert np.allclose(out["ema_slope"].values[10:], 0.0, atol=1e-8)

    def test_ema_slope_positive_in_uptrend(self, trend_df):
        out = MovingAveragesEngine().generate(trend_df)
        assert out["ema_slope"].tail(100).mean() > 0

    def test_ema_slope_units_percent(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        # For normal forex prices ema_slope should be small (< 5%)
        assert out["ema_slope"].abs().max() < 5.0

    def test_ema_cross_dtype(self, rand_df):
        out = MovingAveragesEngine().generate(rand_df)
        assert out["ema_cross"].dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# TestMomentumContract
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumContract:
    def test_registered(self):
        assert "momentum" in FeatureRegistry.all_features()

    def test_name(self):
        assert MomentumEngine.name == "momentum"

    def test_category(self):
        assert MomentumEngine.category == "technical"

    def test_dependencies_empty(self):
        assert MomentumEngine.dependencies == []

    def test_required_columns(self):
        for col in ("high", "low", "close"):
            assert col in MomentumEngine.required_columns

    def test_metadata_output_columns(self):
        eng = MomentumEngine()
        assert len(eng.metadata().output_columns) == 11


# ─────────────────────────────────────────────────────────────────────────────
# TestRSI
# ─────────────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_bounds(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert (out["rsi"] >= 0).all() and (out["rsi"] <= 100).all()

    def test_rsi_constant_price(self):
        n  = 100
        df = pd.DataFrame(
            {"high": np.ones(n), "low": np.ones(n), "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = MomentumEngine().generate(df)
        assert np.allclose(out["rsi"].values[14:], 100.0, atol=1e-8)

    def test_rsi_uptrend_above_50(self, trend_df):
        out = MomentumEngine().generate(trend_df)
        assert out["rsi"].tail(100).mean() > 50

    def test_rsi_downtrend_below_50(self):
        n     = 300
        close = np.linspace(1.3, 1.0, n)
        df    = pd.DataFrame(
            {"high": close + 0.001, "low": close - 0.001, "close": close},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = MomentumEngine().generate(df)
        assert out["rsi"].tail(100).mean() < 50

    def test_rsi_dtype(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert out["rsi"].dtype == np.float64

    def test_rsi_no_nan(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert not out["rsi"].isnull().any()


# ─────────────────────────────────────────────────────────────────────────────
# TestStochastic
# ─────────────────────────────────────────────────────────────────────────────

class TestStochastic:
    def test_stoch_k_bounds(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert (out["stochastic_k"] >= 0).all() and (out["stochastic_k"] <= 100).all()

    def test_stoch_d_bounds(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert (out["stochastic_d"] >= 0).all() and (out["stochastic_d"] <= 100).all()

    def test_stoch_d_smoother_than_k(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert out["stochastic_k"].std() >= out["stochastic_d"].std()

    def test_stoch_constant_at_50(self):
        n  = 50
        close = np.full(n, 1.1)
        df = pd.DataFrame(
            {"high": close, "low": close, "close": close},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = MomentumEngine().generate(df)
        assert np.allclose(out["stochastic_k"].values, 50.0, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# TestMACD
# ─────────────────────────────────────────────────────────────────────────────

class TestMACD:
    def test_histogram_is_macd_minus_signal(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        np.testing.assert_allclose(
            out["macd_histogram"].values,
            out["macd"].values - out["macd_signal"].values,
            rtol=1e-10,
        )

    def test_macd_zero_on_constant(self):
        n  = 100
        df = pd.DataFrame(
            {"high": np.ones(n), "low": np.ones(n), "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = MomentumEngine().generate(df)
        assert np.allclose(out["macd"].values, 0.0, atol=1e-10)

    def test_macd_positive_in_uptrend(self, trend_df):
        out = MomentumEngine().generate(trend_df)
        assert out["macd"].tail(100).mean() > 0

    def test_macd_no_nan(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert not out[["macd", "macd_signal", "macd_histogram"]].isnull().any().any()

    def test_macd_matches_ema_diff(self, rand_df):
        out  = MomentumEngine().generate(rand_df)
        ema12 = rand_df["close"].ewm(span=12, adjust=False).mean().values
        ema26 = rand_df["close"].ewm(span=26, adjust=False).mean().values
        np.testing.assert_allclose(out["macd"].values, ema12 - ema26, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# TestCCIWilliamsROCTSI
# ─────────────────────────────────────────────────────────────────────────────

class TestCCIWilliamsROCTSI:
    def test_williams_r_bounds(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert (out["williams_r"] >= -100).all() and (out["williams_r"] <= 0).all()

    def test_roc_zero_on_constant(self):
        n  = 50
        df = pd.DataFrame(
            {"high": np.ones(n), "low": np.ones(n), "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = MomentumEngine().generate(df)
        assert np.allclose(out["roc"].values[13:], 0.0, atol=1e-8)

    def test_roc_positive_uptrend(self, trend_df):
        out = MomentumEngine().generate(trend_df)
        assert out["roc"].tail(100).mean() > 0

    def test_tsi_bounds(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert (out["tsi"] >= -100).all() and (out["tsi"] <= 100).all()

    def test_tsi_zero_constant(self):
        n  = 100
        df = pd.DataFrame(
            {"high": np.ones(n), "low": np.ones(n), "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = MomentumEngine().generate(df)
        assert np.allclose(out["tsi"].values[30:], 0.0, atol=1e-8)

    def test_cci_zero_on_constant(self):
        n  = 50
        df = pd.DataFrame(
            {"high": np.ones(n), "low": np.ones(n), "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = MomentumEngine().generate(df)
        assert np.allclose(out["cci"].values, 0.0, atol=1e-8)

    def test_momentum_positive_uptrend(self, trend_df):
        out = MomentumEngine().generate(trend_df)
        assert out["price_momentum"].tail(100).mean() > 0

    def test_all_momentum_no_inf(self, rand_df):
        out = MomentumEngine().generate(rand_df)
        assert not np.isinf(out.values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestTrendContract
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendContract:
    def test_registered(self):
        assert "trend" in FeatureRegistry.all_features()

    def test_name(self):
        assert TrendEngine.name == "trend"

    def test_category(self):
        assert TrendEngine.category == "technical"

    def test_dependencies_empty(self):
        assert TrendEngine.dependencies == []

    def test_required_columns(self):
        for col in ("high", "low", "close"):
            assert col in TrendEngine.required_columns

    def test_metadata_output_columns(self):
        assert len(TrendEngine().metadata().output_columns) == 7


# ─────────────────────────────────────────────────────────────────────────────
# TestADXDI
# ─────────────────────────────────────────────────────────────────────────────

class TestADXDI:
    def test_adx_non_negative(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert (out["adx"] >= 0).all()

    def test_di_non_negative(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert (out["plus_di"] >= 0).all()
        assert (out["minus_di"] >= 0).all()

    def test_plus_di_gt_minus_di_uptrend(self, trend_df):
        out = TrendEngine().generate(trend_df)
        assert out["plus_di"].tail(50).mean() > out["minus_di"].tail(50).mean()

    def test_minus_di_gt_plus_di_downtrend(self):
        n     = 300
        close = np.linspace(1.3, 1.0, n)
        df    = pd.DataFrame(
            {"high": close + 0.001, "low": close - 0.001, "close": close},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = TrendEngine().generate(df)
        assert out["minus_di"].tail(50).mean() > out["plus_di"].tail(50).mean()

    def test_adx_no_nan(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert not out["adx"].isnull().any()

    def test_adx_bounded(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert (out["adx"] <= 100).all()

    def test_psar_helper_uptrend_below_price(self, trend_df):
        high  = trend_df["high"].values
        low   = trend_df["low"].values
        psar  = _parabolic_sar(high, low, 0.02, 0.02, 0.20)
        # after warmup in a clean uptrend PSAR should be below high
        assert (psar[50:] < high[50:]).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestAroon
# ─────────────────────────────────────────────────────────────────────────────

class TestAroon:
    def test_aroon_bounds(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert (out["aroon_up"] >= 0).all() and (out["aroon_up"] <= 100).all()
        assert (out["aroon_down"] >= 0).all() and (out["aroon_down"] <= 100).all()

    def test_aroon_oscillator_is_diff(self, rand_df):
        out = TrendEngine().generate(rand_df)
        np.testing.assert_allclose(
            out["aroon_oscillator"].values,
            out["aroon_up"].values - out["aroon_down"].values,
            rtol=1e-10,
        )

    def test_aroon_up_dominant_in_uptrend(self, trend_df):
        out = TrendEngine().generate(trend_df)
        assert out["aroon_up"].tail(100).mean() > out["aroon_down"].tail(100).mean()

    def test_aroon_no_nan(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert not out[["aroon_up", "aroon_down", "aroon_oscillator"]].isnull().any().any()

    def test_aroon_oscillator_range(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert (out["aroon_oscillator"] >= -100).all()
        assert (out["aroon_oscillator"] <= 100).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestParabolicSAR
# ─────────────────────────────────────────────────────────────────────────────

class TestParabolicSAR:
    def test_psar_length(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert len(out["parabolic_sar"]) == len(rand_df)

    def test_psar_no_nan(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert not np.isnan(out["parabolic_sar"].values).any()

    def test_psar_flip_occurs(self, rand_df):
        """PSAR should be on both sides of price over 300 bars."""
        out   = TrendEngine().generate(rand_df)
        close = rand_df["close"].values
        psar  = out["parabolic_sar"].values
        above = (psar > close).any()
        below = (psar < close).any()
        assert above and below

    def test_psar_single_bar(self):
        df = pd.DataFrame(
            {"high": [1.1], "low": [1.0], "close": [1.05]},
            index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"),
        )
        out = TrendEngine().generate(df)
        assert not np.isnan(out["parabolic_sar"].values[0])

    def test_psar_dtype(self, rand_df):
        out = TrendEngine().generate(rand_df)
        assert out["parabolic_sar"].dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# TestVolatilityContract
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityContract:
    def test_registered(self):
        assert "volatility" in FeatureRegistry.all_features()

    def test_name(self):
        assert VolatilityEngine.name == "volatility"

    def test_category(self):
        assert VolatilityEngine.category == "technical"

    def test_dependencies_empty(self):
        assert VolatilityEngine.dependencies == []

    def test_required_columns(self):
        for col in ("high", "low", "close"):
            assert col in VolatilityEngine.required_columns

    def test_metadata_output_columns(self):
        assert len(VolatilityEngine().metadata().output_columns) == 11


# ─────────────────────────────────────────────────────────────────────────────
# TestATR
# ─────────────────────────────────────────────────────────────────────────────

class TestATR:
    def test_atr_non_negative(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert (out["atr"] >= 0).all()

    def test_normalized_atr_non_negative(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert (out["normalized_atr"] >= 0).all()

    def test_atr_increases_with_range(self):
        n   = 100
        # Wide range bars
        wide = pd.DataFrame(
            {"high": np.ones(n) * 1.2, "low": np.ones(n) * 0.8, "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        # Narrow range bars
        narrow = pd.DataFrame(
            {"high": np.ones(n) * 1.01, "low": np.ones(n) * 0.99, "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out_w = VolatilityEngine().generate(wide)
        out_n = VolatilityEngine().generate(narrow)
        assert out_w["atr"].tail(50).mean() > out_n["atr"].tail(50).mean()

    def test_atr_no_nan(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert not out["atr"].isnull().any()

    def test_norm_atr_reasonable(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        # For forex ~0.0001–0.01 pip ranges; normalized should be < 5%
        assert out["normalized_atr"].mean() < 5.0


# ─────────────────────────────────────────────────────────────────────────────
# TestBollingerBands
# ─────────────────────────────────────────────────────────────────────────────

class TestBollingerBands:
    def test_upper_ge_lower(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert (out["bb_upper"] >= out["bb_lower"]).all()

    def test_percent_b_in_range_mostly(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        # 95%+ of bars should be within [−0.5, 1.5] — only extreme moves fall outside
        within = ((out["bb_percent_b"] > -0.5) & (out["bb_percent_b"] < 1.5)).mean()
        assert within > 0.95

    def test_bb_width_zero_constant(self):
        n  = 50
        df = pd.DataFrame(
            {"high": np.ones(n), "low": np.ones(n), "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = VolatilityEngine().generate(df)
        # std dev = 0 → bands collapse → bb_width = 0
        assert np.allclose(out["bb_width"].values[19:], 0.0, atol=1e-8)

    def test_bb_no_nan(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        cols = ["bb_upper", "bb_lower", "bb_width", "bb_percent_b"]
        assert not out[cols].isnull().any().any()

    def test_upper_bounds_close(self, rand_df):
        """Close should rarely be above the upper band by definition."""
        out   = VolatilityEngine().generate(rand_df)
        close = rand_df["close"].values
        # rolling std can lag; just check the tail after warmup
        assert (close[30:] <= out["bb_upper"].values[30:] * 1.001).mean() > 0.85

    def test_bb_width_positive_on_noise(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert (out["bb_width"].tail(200) > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestKeltnerDonchian
# ─────────────────────────────────────────────────────────────────────────────

class TestKeltnerDonchian:
    def test_kc_upper_ge_lower(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert (out["kc_upper"] >= out["kc_lower"]).all()

    def test_dc_upper_ge_lower(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert (out["dc_upper"] >= out["dc_lower"]).all()

    def test_dc_contains_close(self, rand_df):
        out   = VolatilityEngine().generate(rand_df)
        close = rand_df["close"].values
        assert (close[19:] <= out["dc_upper"].values[19:] + 1e-10).all()
        assert (close[19:] >= out["dc_lower"].values[19:] - 1e-10).all()

    def test_kc_no_nan(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert not out[["kc_upper", "kc_lower"]].isnull().any().any()

    def test_dc_no_nan(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert not out[["dc_upper", "dc_lower"]].isnull().any().any()


# ─────────────────────────────────────────────────────────────────────────────
# TestChaikinVolatility
# ─────────────────────────────────────────────────────────────────────────────

class TestChaikinVolatility:
    def test_chaikin_zero_constant(self):
        n  = 50
        df = pd.DataFrame(
            {"high": np.ones(n) * 1.01, "low": np.ones(n) * 0.99, "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = VolatilityEngine().generate(df)
        assert np.allclose(out["chaikin_volatility"].values[20:], 0.0, atol=1e-8)

    def test_chaikin_no_nan(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert not out["chaikin_volatility"].isnull().any()

    def test_chaikin_no_inf(self, rand_df):
        out = VolatilityEngine().generate(rand_df)
        assert not np.isinf(out["chaikin_volatility"].values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestOscillatorContract
# ─────────────────────────────────────────────────────────────────────────────

class TestOscillatorContract:
    def test_registered(self):
        assert "oscillators" in FeatureRegistry.all_features()

    def test_name(self):
        assert OscillatorsEngine.name == "oscillators"

    def test_category(self):
        assert OscillatorsEngine.category == "technical"

    def test_dependencies_empty(self):
        assert OscillatorsEngine.dependencies == []

    def test_required_columns(self):
        for col in ("high", "low", "close", "volume"):
            assert col in OscillatorsEngine.required_columns

    def test_metadata_output_columns(self):
        assert len(OscillatorsEngine().metadata().output_columns) == 8


# ─────────────────────────────────────────────────────────────────────────────
# TestVWAP
# ─────────────────────────────────────────────────────────────────────────────

class TestVWAP:
    def test_vwap_no_nan(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not out["vwap"].isnull().any()

    def test_vwap_resets_daily(self):
        """VWAP on day 2 bar 0 should equal that bar's typical price (vol=const)."""
        # Constant volume → VWAP = arithmetic mean of TP over the day
        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        n   = len(idx)
        tp  = np.arange(1.0, n + 1.0) * 0.001 + 1.0
        df  = pd.DataFrame(
            {"high": tp + 0.0001, "low": tp - 0.0001,
             "close": tp, "volume": np.ones(n) * 1000.0},
            index=idx,
        )
        out = OscillatorsEngine().generate(df)
        # Bar 24 is the first bar of day 2; VWAP should reset to that bar's TP
        tp24 = (df["high"].iloc[24] + df["low"].iloc[24] + df["close"].iloc[24]) / 3.0
        assert abs(out["vwap"].iloc[24] - tp24) < 1e-6

    def test_vwap_between_low_and_high(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert (out["vwap"] >= rand_df["low"].min() * 0.99).all()
        assert (out["vwap"] <= rand_df["high"].max() * 1.01).all()

    def test_vwma_no_nan(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not out["vwma"].isnull().any()

    def test_vwma_close_to_price(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        corr = np.corrcoef(rand_df["close"].values, out["vwma"].values)[0, 1]
        assert corr > 0.90


# ─────────────────────────────────────────────────────────────────────────────
# TestOBVAD
# ─────────────────────────────────────────────────────────────────────────────

class TestOBVAD:
    def test_obv_increases_in_uptrend(self, trend_df):
        out = OscillatorsEngine().generate(trend_df)
        # In a rising trend OBV should be monotonically non-decreasing
        diff = np.diff(out["obv"].values)
        assert (diff >= 0).all()

    def test_obv_no_nan(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not out["obv"].isnull().any()

    def test_ad_no_nan(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not out["ad"].isnull().any()

    def test_ad_sign_matches_clv(self, rand_df):
        """AD should increase when CLV > 0 (close near high)."""
        close = rand_df["close"].values
        high  = rand_df["high"].values
        low   = rand_df["low"].values
        hl    = high - low
        clv   = np.where(hl > 0, ((close - low) - (high - close)) / hl, 0.0)
        out   = OscillatorsEngine().generate(rand_df)
        ad    = out["ad"].values
        # Both series start at the same sign pattern
        ad_diff  = np.sign(np.diff(ad))
        clv_sign = np.sign(clv[1:])
        match    = (ad_diff == clv_sign).mean()
        assert match > 0.7


# ─────────────────────────────────────────────────────────────────────────────
# TestCMFMFI
# ─────────────────────────────────────────────────────────────────────────────

class TestCMFMFI:
    def test_cmf_bounds(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert (out["cmf"] >= -1.0).all() and (out["cmf"] <= 1.0).all()

    def test_mfi_bounds(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert (out["mfi"] >= 0).all() and (out["mfi"] <= 100).all()

    def test_cmf_no_nan(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not out["cmf"].isnull().any()

    def test_mfi_no_nan(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not out["mfi"].isnull().any()

    def test_cmf_zero_on_midpoint_close(self):
        """When close = (high+low)/2 CLV=0 → CMF=0."""
        n   = 50
        low = np.ones(n) * 1.0
        hi  = np.ones(n) * 1.1
        mid = (low + hi) / 2.0
        df  = pd.DataFrame(
            {"high": hi, "low": low, "close": mid, "volume": np.ones(n) * 100},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = OscillatorsEngine().generate(df)
        assert np.allclose(out["cmf"].values, 0.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# TestForceIndexEOM
# ─────────────────────────────────────────────────────────────────────────────

class TestForceIndexEOM:
    def test_force_index_positive_uptrend(self, trend_df):
        out = OscillatorsEngine().generate(trend_df)
        assert out["force_index"].tail(100).mean() > 0

    def test_force_index_no_nan(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not out["force_index"].isnull().any()

    def test_eom_no_nan(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not out["eom"].isnull().any()

    def test_eom_no_inf(self, rand_df):
        out = OscillatorsEngine().generate(rand_df)
        assert not np.isinf(out["eom"].values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestTechnicalEngineContract
# ─────────────────────────────────────────────────────────────────────────────

class TestTechnicalEngineContract:
    def test_registered(self):
        assert "technical" in FeatureRegistry.all_features()

    def test_name(self):
        assert TechnicalEngine.name == "technical"

    def test_category(self):
        assert TechnicalEngine.category == "technical"

    def test_dependencies(self):
        assert set(TechnicalEngine.dependencies) == {
            "moving_averages", "momentum", "trend", "volatility", "oscillators"
        }

    def test_required_columns(self):
        for col in ("close", "ema200", "vwap", "macd", "atr",
                    "rsi", "stochastic_k", "adx", "plus_di", "minus_di"):
            assert col in TechnicalEngine.required_columns

    def test_metadata_output_columns(self):
        assert len(TechnicalEngine().metadata().output_columns) == 5


def _build_combined_df(rand_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate the pipeline's running_df with all upstream columns present."""
    ma  = MovingAveragesEngine().generate(rand_df)
    mom = MomentumEngine().generate(rand_df)
    tr  = TrendEngine().generate(rand_df)
    vol = VolatilityEngine().generate(rand_df)
    osc = OscillatorsEngine().generate(rand_df)
    return pd.concat([rand_df, ma, mom, tr, vol, osc], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# TestCrossIndicator
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossIndicator:
    def test_price_vs_ema200_positive_uptrend(self, trend_df):
        combined = _build_combined_df(trend_df)
        out = TechnicalEngine().generate(combined)
        assert out["price_vs_ema200"].tail(50).mean() > 0

    def test_price_vs_vwap_direction(self, rand_df):
        combined = _build_combined_df(rand_df)
        out = TechnicalEngine().generate(combined)
        # price_vs_vwap should be ≠ 0 most of the time
        nonzero = (out["price_vs_vwap"].abs() > 0).mean()
        assert nonzero > 0.5

    def test_macd_normalized_bounded(self, rand_df):
        combined = _build_combined_df(rand_df)
        out = TechnicalEngine().generate(combined)
        # Normalised by ATR it should rarely exceed ±10
        assert out["macd_normalized"].abs().max() < 50.0

    def test_trend_strength_range(self, rand_df):
        combined = _build_combined_df(rand_df)
        out = TechnicalEngine().generate(combined)
        assert (out["trend_strength"] >= -1.0).all()
        assert (out["trend_strength"] <= 1.0).all()

    def test_trend_strength_positive_uptrend(self, trend_df):
        combined = _build_combined_df(trend_df)
        out = TechnicalEngine().generate(combined)
        assert out["trend_strength"].tail(50).mean() > 0

    def test_rsi_stoch_divergence_dtype(self, rand_df):
        combined = _build_combined_df(rand_df)
        out = TechnicalEngine().generate(combined)
        assert out["rsi_stoch_divergence"].dtype == np.float64

    def test_all_cross_no_nan(self, rand_df):
        combined = _build_combined_df(rand_df)
        out = TechnicalEngine().generate(combined)
        assert not out.isnull().any().any()


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_row(self):
        df = pd.DataFrame(
            {"open": [1.1], "high": [1.11], "low": [1.09],
             "close": [1.105], "volume": [1000.0]},
            index=pd.DatetimeIndex(["2024-01-01 00:00+00:00"]),
        )
        for Cls in (MovingAveragesEngine, MomentumEngine,
                    TrendEngine, VolatilityEngine, OscillatorsEngine):
            out = Cls().generate(df)
            assert len(out) == 1
            assert not np.isnan(out.values).any(), f"{Cls.__name__} produced NaN on single row"

    def test_zero_volume(self):
        n  = 50
        df = pd.DataFrame(
            {"high": np.linspace(1.1, 1.2, n),
             "low":  np.linspace(1.0, 1.1, n),
             "close": np.linspace(1.05, 1.15, n),
             "volume": np.zeros(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = OscillatorsEngine().generate(df)
        assert not np.isnan(out.values).any()
        assert not np.isinf(out.values).any()

    def test_constant_price_no_nan_momentum(self):
        n  = 100
        df = pd.DataFrame(
            {"high": np.ones(n), "low": np.ones(n), "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = MomentumEngine().generate(df)
        assert not np.isnan(out.values).any()

    def test_constant_price_no_inf_volatility(self):
        n  = 50
        df = pd.DataFrame(
            {"high": np.ones(n) * 1.01, "low": np.ones(n) * 0.99,
             "close": np.ones(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        out = VolatilityEngine().generate(df)
        assert not np.isinf(out.values).any()

    def test_two_rows_psar(self):
        df = pd.DataFrame(
            {"high": [1.1, 1.2], "low": [1.0, 1.1], "close": [1.05, 1.15]},
            index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        )
        out = TrendEngine().generate(df)
        assert len(out["parabolic_sar"]) == 2
        assert not np.isnan(out["parabolic_sar"].values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestIntegrationAndDtype
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationAndDtype:
    def test_all_engines_float64(self, rand_df):
        for Cls in (MovingAveragesEngine, MomentumEngine,
                    TrendEngine, VolatilityEngine, OscillatorsEngine):
            out = Cls().generate(rand_df)
            bad = [c for c in out.columns if out[c].dtype != np.float64]
            assert bad == [], f"{Cls.__name__}: non-float64 cols: {bad}"

    def test_composite_engine_float64(self, rand_df):
        combined = _build_combined_df(rand_df)
        out = TechnicalEngine().generate(combined)
        assert (out.dtypes == np.float64).all()

    def test_all_registry_names_present(self):
        registry = FeatureRegistry.all_features()
        for name in ("moving_averages", "momentum", "trend", "volatility",
                     "oscillators", "technical"):
            assert name in registry, f"'{name}' not registered"

    def test_total_output_columns(self, rand_df):
        """Combined output across all 5 sub-engines = 12+11+7+11+8 = 49 cols."""
        combined = 0
        for Cls in (MovingAveragesEngine, MomentumEngine,
                    TrendEngine, VolatilityEngine, OscillatorsEngine):
            combined += len(Cls().generate(rand_df).columns)
        assert combined == 49

    def test_composite_adds_5_cols(self, rand_df):
        combined_df = _build_combined_df(rand_df)
        out = TechnicalEngine().generate(combined_df)
        assert out.shape[1] == 5


# ─────────────────────────────────────────────────────────────────────────────
# TestPerformance
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:
    def test_all_engines_under_10s_on_10k_bars(self):
        df = _make_ohlcv(n=10_000)
        t0 = time.perf_counter()
        ma  = MovingAveragesEngine().generate(df)
        mom = MomentumEngine().generate(df)
        tr  = TrendEngine().generate(df)
        vol = VolatilityEngine().generate(df)
        osc = OscillatorsEngine().generate(df)
        combined = pd.concat([df, ma, mom, tr, vol, osc], axis=1)
        TechnicalEngine().generate(combined)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Full technical engine took {elapsed:.2f}s on 10k bars"
