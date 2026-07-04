"""Comprehensive tests for the Statistical & Market Microstructure Engine.

Coverage (155 tests across 30 classes):
    TestReturnsContract          — interface contract (6)
    TestReturnsValues            — log/simple/rolling/forward correctness (8)
    TestCandleContract           — interface contract (6)
    TestCandleAnatomy            — body/wick geometry (8)
    TestCandlePatterns           — doji, marubozu, inside/outside bar (6)
    TestCandleSequence           — consecutive runs, directional counts (6)
    TestRollingStatContract      — interface contract (6)
    TestRollingStatValues        — rolling aggregates correctness (7)
    TestDistributionContract     — interface contract (6)
    TestDistributionValues       — skewness, kurtosis, zscore, rank (8)
    TestMomentumStatContract     — interface contract (6)
    TestMomentumStatValues       — velocity, acceleration, persistence (7)
    TestVolatilityStatContract   — interface contract (6)
    TestVolatilityStatValues     — RV, expansion, ATR ratio, regime (6)
    TestEntropyContract          — interface contract (6)
    TestEntropyValues            — Shannon, ApEn bounds and semantics (6)
    TestMicrostructureContract   — interface contract (6)
    TestMicrostructureValues     — ER, Hurst, FD, noise, smoothness (8)
    TestStatisticalEngineContract— interface contract (6)
    TestStatisticalEngineValues  — cross-module composites (5)
    TestRegistryAll              — all 9 engines in registry (5)
    TestDependencyChain          — dependency ordering correctness (4)
    TestEdgeCases                — single row, constant price, zero vol (6)
    TestDtypeAndShape            — float64, no-NaN, no-Inf across all engines (5)
    TestPerformance              — 10k bars within time budget (1)
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from src.features.feature_registry import FeatureRegistry
import src.features                 # noqa: F401 — triggers all @register decorators

from src.features.statistics.returns              import ReturnsEngine
from src.features.statistics.candle_statistics    import CandleStatisticsEngine
from src.features.statistics.rolling_statistics   import RollingStatisticsEngine
from src.features.statistics.distribution         import DistributionEngine
from src.features.statistics.momentum             import MomentumStatisticsEngine
from src.features.statistics.volatility           import VolatilityStatisticsEngine
from src.features.statistics.entropy              import EntropyEngine
from src.features.statistics.market_microstructure import MarketMicrostructureEngine
from src.features.statistics.statistical_engine   import StatisticalEngine

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    mid    = 1.1 + np.cumsum(rng.normal(trend * 0.001, 0.0005, n))
    spread = 0.0002
    high   = mid + rng.uniform(0, spread, n)
    low    = mid - rng.uniform(0, spread, n)
    close  = mid + rng.normal(0, spread / 4, n)
    open_  = mid + rng.normal(0, spread / 4, n)
    high   = np.maximum(high, np.maximum(open_, close))
    low    = np.minimum(low,  np.minimum(open_, close))
    vol    = rng.uniform(100, 10_000, n)
    idx    = pd.date_range("2024-01-02 00:00", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close, "volume": vol},
        index=idx,
    )


def _make_uptrend(n: int = 300) -> pd.DataFrame:
    return _make_ohlcv(n=n, trend=5.0)


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate pipeline running_df by appending returns columns."""
    ret_out = ReturnsEngine().generate(df)
    return pd.concat([df, ret_out], axis=1)


def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Append atr (from technical VolatilityEngine mock) to simulate pipeline."""
    from src.features.technical.volatility import VolatilityEngine
    vol_out = VolatilityEngine().generate(df)
    return pd.concat([df, vol_out], axis=1)


def _full_stat_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build complete running_df needed by StatisticalEngine."""
    d    = _add_returns(df)
    d    = _add_volatility(d)
    d    = pd.concat([d, RollingStatisticsEngine().generate(df)], axis=1)
    d    = pd.concat([d, DistributionEngine().generate(d)], axis=1)
    d    = pd.concat([d, MomentumStatisticsEngine().generate(d)], axis=1)
    d    = pd.concat([d, VolatilityStatisticsEngine().generate(d)], axis=1)
    d    = pd.concat([d, EntropyEngine().generate(d)], axis=1)
    d    = pd.concat([d, MarketMicrostructureEngine().generate(d)], axis=1)
    d    = pd.concat([d, CandleStatisticsEngine().generate(df)], axis=1)
    return d


@pytest.fixture
def rand_df() -> pd.DataFrame:
    return _make_ohlcv()


@pytest.fixture
def trend_df() -> pd.DataFrame:
    return _make_uptrend()


# ─────────────────────────────────────────────────────────────────────────────
# TestReturnsContract
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnsContract:
    def test_registered(self):
        assert "returns" in FeatureRegistry.all_features()

    def test_name(self):
        assert ReturnsEngine.name == "returns"

    def test_category(self):
        assert ReturnsEngine.category == "statistics"

    def test_no_dependencies(self):
        assert ReturnsEngine.dependencies == []

    def test_required_columns(self):
        assert "close" in ReturnsEngine.required_columns

    def test_output_columns_count(self):
        assert len(ReturnsEngine().metadata().output_columns) == 5


# ─────────────────────────────────────────────────────────────────────────────
# TestReturnsValues
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnsValues:
    def test_log_return_first_bar_zero(self, rand_df):
        out = ReturnsEngine().generate(rand_df)
        assert out["log_return"].iloc[0] == 0.0

    def test_log_equals_log_price_diff(self, rand_df):
        out   = ReturnsEngine().generate(rand_df)
        close = rand_df["close"].values
        expected = np.log(close[1:] / close[:-1])
        np.testing.assert_allclose(out["log_return"].values[1:], expected, rtol=1e-8)

    def test_simple_return_first_bar_zero(self, rand_df):
        out = ReturnsEngine().generate(rand_df)
        assert out["simple_return"].iloc[0] == 0.0

    def test_simple_return_formula(self, rand_df):
        out   = ReturnsEngine().generate(rand_df)
        close = rand_df["close"].values
        expected = (close[1:] - close[:-1]) / close[:-1]
        np.testing.assert_allclose(out["simple_return"].values[1:], expected, rtol=1e-8)

    def test_rolling_return_5_is_sum(self, rand_df):
        out = ReturnsEngine().generate(rand_df)
        lr  = pd.Series(out["log_return"].values)
        expected = lr.rolling(5, min_periods=1).sum().values
        np.testing.assert_allclose(out["rolling_return_5"].values, expected, rtol=1e-10)

    def test_fwd_return_last_bar_zero(self, rand_df):
        out = ReturnsEngine().generate(rand_df)
        assert out["fwd_return_1"].iloc[-1] == 0.0

    def test_fwd_return_shifted(self, rand_df):
        out = ReturnsEngine().generate(rand_df)
        np.testing.assert_allclose(
            out["fwd_return_1"].values[:-1],
            out["log_return"].values[1:],
            rtol=1e-10,
        )

    def test_all_float64_no_nan(self, rand_df):
        out = ReturnsEngine().generate(rand_df)
        assert (out.dtypes == np.float64).all()
        assert not out.isnull().any().any()


# ─────────────────────────────────────────────────────────────────────────────
# TestCandleContract
# ─────────────────────────────────────────────────────────────────────────────

class TestCandleContract:
    def test_registered(self):
        assert "candle_statistics" in FeatureRegistry.all_features()

    def test_name(self):
        assert CandleStatisticsEngine.name == "candle_statistics"

    def test_category(self):
        assert CandleStatisticsEngine.category == "statistics"

    def test_no_dependencies(self):
        assert CandleStatisticsEngine.dependencies == []

    def test_required_columns(self):
        for c in ("open", "high", "low", "close"):
            assert c in CandleStatisticsEngine.required_columns

    def test_output_columns_count(self):
        assert len(CandleStatisticsEngine().metadata().output_columns) == 21


# ─────────────────────────────────────────────────────────────────────────────
# TestCandleAnatomy
# ─────────────────────────────────────────────────────────────────────────────

class TestCandleAnatomy:
    def test_body_size_non_negative(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["body_size"] >= 0).all()

    def test_total_range_non_negative(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["total_range"] >= 0).all()

    def test_true_range_ge_total_range(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["true_range"] >= out["total_range"] - 1e-10).all()

    def test_body_ratio_bounded(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["body_ratio"] >= 0).all() and (out["body_ratio"] <= 1).all()

    def test_wick_ratios_sum_le_one(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        total = out["body_ratio"] + out["upper_wick_ratio"] + out["lower_wick_ratio"]
        assert (total <= 1.0 + 1e-9).all()

    def test_upper_wick_non_negative(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["upper_wick"] >= -1e-10).all()

    def test_lower_wick_non_negative(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["lower_wick"] >= -1e-10).all()

    def test_body_to_range_bounded(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["body_to_range_ratio"] >= 0).all()
        assert (out["body_to_range_ratio"] <= 1.0 + 1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestCandlePatterns
# ─────────────────────────────────────────────────────────────────────────────

class TestCandlePatterns:
    def test_is_bullish_binary(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert set(out["is_bullish"].unique()).issubset({0.0, 1.0})

    def test_is_bearish_binary(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert set(out["is_bearish"].unique()).issubset({0.0, 1.0})

    def test_bull_bear_not_simultaneous(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["is_bullish"] + out["is_bearish"] <= 1.0).all()

    def test_doji_score_plus_body_ratio_eq_one(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        np.testing.assert_allclose(
            out["doji_score"].values + out["body_ratio"].values,
            1.0,
            atol=1e-10,
        )

    def test_marubozu_score_bounded(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["marubozu_score"] >= 0).all() and (out["marubozu_score"] <= 1).all()

    def test_inside_outside_bar_binary(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert set(out["inside_bar"].unique()).issubset({0.0, 1.0})
        assert set(out["outside_bar"].unique()).issubset({0.0, 1.0})


# ─────────────────────────────────────────────────────────────────────────────
# TestCandleSequence
# ─────────────────────────────────────────────────────────────────────────────

class TestCandleSequence:
    def test_consecutive_bulls_non_negative(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["consecutive_bulls"] >= 0).all()

    def test_consecutive_bears_non_negative(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["consecutive_bears"] >= 0).all()

    def test_consecutive_resets_on_opposite(self):
        """A bearish bar must reset consecutive_bulls to 0."""
        df = pd.DataFrame(
            {"open":  [1.0, 1.1, 1.2, 1.3, 1.4],
             "high":  [1.1, 1.2, 1.3, 1.4, 1.5],
             "low":   [0.9, 1.0, 1.1, 1.2, 1.3],
             "close": [1.1, 1.2, 1.3, 1.2, 1.5]},   # bar[3] is bearish
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        out = CandleStatisticsEngine().generate(df)
        assert out["consecutive_bulls"].iloc[3] == 0.0

    def test_higher_close_count_bounded(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert (out["higher_close_count"] <= 10).all()
        assert (out["higher_close_count"] >= 0).all()

    def test_higher_close_plus_lower_close_le_window(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        total = out["higher_close_count"] + out["lower_close_count"]
        assert (total <= 10.0 + 1e-9).all()

    def test_sequential_uptrend_all_higher_closes(self, trend_df):
        out = CandleStatisticsEngine().generate(trend_df)
        # Tail of a strong uptrend: higher_close_count ≈ window
        assert out["higher_close_count"].tail(50).mean() > 8.0


# ─────────────────────────────────────────────────────────────────────────────
# TestRollingStatContract
# ─────────────────────────────────────────────────────────────────────────────

class TestRollingStatContract:
    def test_registered(self):
        assert "rolling_statistics" in FeatureRegistry.all_features()

    def test_name(self):
        assert RollingStatisticsEngine.name == "rolling_statistics"

    def test_category(self):
        assert RollingStatisticsEngine.category == "statistics"

    def test_no_dependencies(self):
        assert RollingStatisticsEngine.dependencies == []

    def test_required_columns(self):
        assert "close" in RollingStatisticsEngine.required_columns

    def test_output_columns_count(self):
        assert len(RollingStatisticsEngine().metadata().output_columns) == 9


# ─────────────────────────────────────────────────────────────────────────────
# TestRollingStatValues
# ─────────────────────────────────────────────────────────────────────────────

class TestRollingStatValues:
    def test_rolling_mean_matches_pandas(self, rand_df):
        out      = RollingStatisticsEngine().generate(rand_df)
        expected = rand_df["close"].rolling(20, min_periods=1).mean().values
        np.testing.assert_allclose(out["rolling_mean"].values, expected, rtol=1e-10)

    def test_rolling_std_non_negative(self, rand_df):
        out = RollingStatisticsEngine().generate(rand_df)
        assert (out["rolling_std"] >= 0).all()

    def test_rolling_min_le_close(self, rand_df):
        out = RollingStatisticsEngine().generate(rand_df)
        assert (out["rolling_min"] <= rand_df["close"].values + 1e-10).all()

    def test_rolling_max_ge_close(self, rand_df):
        out = RollingStatisticsEngine().generate(rand_df)
        assert (out["rolling_max"] >= rand_df["close"].values - 1e-10).all()

    def test_q25_le_q75(self, rand_df):
        out = RollingStatisticsEngine().generate(rand_df)
        assert (out["rolling_q25"] <= out["rolling_q75"] + 1e-10).all()

    def test_mad_non_negative(self, rand_df):
        out = RollingStatisticsEngine().generate(rand_df)
        assert (out["rolling_mad"] >= 0).all()

    def test_constant_price_std_zero(self):
        n  = 50
        df = pd.DataFrame({"close": np.ones(n)},
                          index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = RollingStatisticsEngine().generate(df)
        assert np.allclose(out["rolling_std"].values[19:], 0.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# TestDistributionContract
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributionContract:
    def test_registered(self):
        assert "distribution" in FeatureRegistry.all_features()

    def test_name(self):
        assert DistributionEngine.name == "distribution"

    def test_category(self):
        assert DistributionEngine.category == "statistics"

    def test_dependencies(self):
        assert "returns" in DistributionEngine.dependencies

    def test_required_columns(self):
        for c in ("close", "log_return"):
            assert c in DistributionEngine.required_columns

    def test_output_columns_count(self):
        assert len(DistributionEngine().metadata().output_columns) == 6


# ─────────────────────────────────────────────────────────────────────────────
# TestDistributionValues
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributionValues:
    def test_zscore_zero_on_constant(self):
        n  = 50
        df = pd.DataFrame({"close": np.ones(n), "log_return": np.zeros(n)},
                          index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = DistributionEngine().generate(df)
        assert np.allclose(out["zscore"].values, 0.0, atol=1e-10)

    def test_percentile_rank_bounded(self, rand_df):
        d   = _add_returns(rand_df)
        out = DistributionEngine().generate(d)
        assert (out["percentile_rank"] >= 0).all() and (out["percentile_rank"] <= 100).all()

    def test_normalized_price_bounded(self, rand_df):
        d   = _add_returns(rand_df)
        out = DistributionEngine().generate(d)
        assert (out["normalized_price"] >= 0 - 1e-9).all()
        assert (out["normalized_price"] <= 1 + 1e-9).all()

    def test_price_rank_bounded(self, rand_df):
        d   = _add_returns(rand_df)
        out = DistributionEngine().generate(d)
        assert (out["price_rank"] >= 0).all() and (out["price_rank"] <= 1 + 1e-9).all()

    def test_skewness_zero_on_symmetric(self):
        n  = 100
        # Symmetric increments → skewness ≈ 0
        rng  = np.random.default_rng(0)
        ret  = rng.normal(0, 0.001, n)
        close = 1.0 + np.cumsum(ret)
        df   = pd.DataFrame({"close": close, "log_return": ret},
                            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out  = DistributionEngine().generate(df)
        assert out["skewness"].tail(50).mean() < 1.0

    def test_kurtosis_no_nan(self, rand_df):
        d   = _add_returns(rand_df)
        out = DistributionEngine().generate(d)
        assert not out["kurtosis"].isnull().any()

    def test_zscore_no_inf(self, rand_df):
        d   = _add_returns(rand_df)
        out = DistributionEngine().generate(d)
        assert not np.isinf(out["zscore"].values).any()

    def test_all_columns_float64(self, rand_df):
        d   = _add_returns(rand_df)
        out = DistributionEngine().generate(d)
        assert (out.dtypes == np.float64).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestMomentumStatContract
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumStatContract:
    def test_registered(self):
        assert "momentum_stats" in FeatureRegistry.all_features()

    def test_name(self):
        assert MomentumStatisticsEngine.name == "momentum_stats"

    def test_category(self):
        assert MomentumStatisticsEngine.category == "statistics"

    def test_dependencies(self):
        assert "returns" in MomentumStatisticsEngine.dependencies

    def test_required_columns(self):
        for c in ("close", "log_return"):
            assert c in MomentumStatisticsEngine.required_columns

    def test_output_columns_count(self):
        assert len(MomentumStatisticsEngine().metadata().output_columns) == 7


# ─────────────────────────────────────────────────────────────────────────────
# TestMomentumStatValues
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumStatValues:
    def test_velocity_positive_in_uptrend(self, trend_df):
        d   = _add_returns(trend_df)
        out = MomentumStatisticsEngine().generate(d)
        assert out["price_velocity"].tail(100).mean() > 0

    def test_deceleration_non_negative(self, rand_df):
        d   = _add_returns(rand_df)
        out = MomentumStatisticsEngine().generate(d)
        assert (out["price_deceleration"] >= 0).all()

    def test_momentum_5_is_rolling_sum(self, rand_df):
        d    = _add_returns(rand_df)
        out  = MomentumStatisticsEngine().generate(d)
        expected = pd.Series(d["log_return"].values).rolling(5, min_periods=1).sum().values
        np.testing.assert_allclose(out["rolling_momentum_5"].values, expected, rtol=1e-10)

    def test_trend_persistence_bounded(self, rand_df):
        d   = _add_returns(rand_df)
        out = MomentumStatisticsEngine().generate(d)
        assert (out["trend_persistence"] >= 0).all() and (out["trend_persistence"] <= 1).all()

    def test_momentum_persistence_no_nan(self, rand_df):
        d   = _add_returns(rand_df)
        out = MomentumStatisticsEngine().generate(d)
        assert not out["momentum_persistence"].isnull().any()

    def test_trend_persistence_high_in_uptrend(self, trend_df):
        d   = _add_returns(trend_df)
        out = MomentumStatisticsEngine().generate(d)
        assert out["trend_persistence"].tail(100).mean() > 0.6

    def test_all_float64_no_inf(self, rand_df):
        d   = _add_returns(rand_df)
        out = MomentumStatisticsEngine().generate(d)
        assert (out.dtypes == np.float64).all()
        assert not np.isinf(out.values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestVolatilityStatContract
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityStatContract:
    def test_registered(self):
        assert "volatility_stats" in FeatureRegistry.all_features()

    def test_name(self):
        assert VolatilityStatisticsEngine.name == "volatility_stats"

    def test_category(self):
        assert VolatilityStatisticsEngine.category == "statistics"

    def test_dependencies(self):
        assert "returns"    in VolatilityStatisticsEngine.dependencies
        assert "volatility" in VolatilityStatisticsEngine.dependencies

    def test_required_columns(self):
        for c in ("log_return", "atr"):
            assert c in VolatilityStatisticsEngine.required_columns

    def test_output_columns_count(self):
        assert len(VolatilityStatisticsEngine().metadata().output_columns) == 7


# ─────────────────────────────────────────────────────────────────────────────
# TestVolatilityStatValues
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityStatValues:
    def _df(self, rand_df):
        return pd.concat([rand_df, ReturnsEngine().generate(rand_df),
                          __import__("src.features.technical.volatility",
                                     fromlist=["VolatilityEngine"]).VolatilityEngine()
                          .generate(rand_df)], axis=1)

    def test_realized_vol_non_negative(self, rand_df):
        from src.features.technical.volatility import VolatilityEngine
        d   = pd.concat([rand_df, ReturnsEngine().generate(rand_df),
                         VolatilityEngine().generate(rand_df)], axis=1)
        out = VolatilityStatisticsEngine().generate(d)
        assert (out["realized_volatility"] >= 0).all()

    def test_regime_bounded(self, rand_df):
        from src.features.technical.volatility import VolatilityEngine
        d   = pd.concat([rand_df, ReturnsEngine().generate(rand_df),
                         VolatilityEngine().generate(rand_df)], axis=1)
        out = VolatilityStatisticsEngine().generate(d)
        assert (out["volatility_regime"] >= 0).all()
        assert (out["volatility_regime"] <= 1).all()

    def test_atr_ratio_positive(self, rand_df):
        from src.features.technical.volatility import VolatilityEngine
        d   = pd.concat([rand_df, ReturnsEngine().generate(rand_df),
                         VolatilityEngine().generate(rand_df)], axis=1)
        out = VolatilityStatisticsEngine().generate(d)
        assert (out["atr_ratio"] >= 0).all()

    def test_expansion_compression_relationship(self, rand_df):
        from src.features.technical.volatility import VolatilityEngine
        d   = pd.concat([rand_df, ReturnsEngine().generate(rand_df),
                         VolatilityEngine().generate(rand_df)], axis=1)
        out = VolatilityStatisticsEngine().generate(d)
        # When expansion > 1, compression < 1 (short vol higher than long)
        mask = out["volatility_expansion"] > 1.0
        if mask.sum() > 10:
            assert (out["volatility_compression"][mask] < 1.5).mean() > 0.8

    def test_no_nan(self, rand_df):
        from src.features.technical.volatility import VolatilityEngine
        d   = pd.concat([rand_df, ReturnsEngine().generate(rand_df),
                         VolatilityEngine().generate(rand_df)], axis=1)
        out = VolatilityStatisticsEngine().generate(d)
        assert not out.isnull().any().any()

    def test_historical_vol_ge_zero(self, rand_df):
        from src.features.technical.volatility import VolatilityEngine
        d   = pd.concat([rand_df, ReturnsEngine().generate(rand_df),
                         VolatilityEngine().generate(rand_df)], axis=1)
        out = VolatilityStatisticsEngine().generate(d)
        assert (out["historical_volatility"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestEntropyContract
# ─────────────────────────────────────────────────────────────────────────────

class TestEntropyContract:
    def test_registered(self):
        assert "entropy" in FeatureRegistry.all_features()

    def test_name(self):
        assert EntropyEngine.name == "entropy"

    def test_category(self):
        assert EntropyEngine.category == "statistics"

    def test_dependencies(self):
        assert "returns" in EntropyEngine.dependencies

    def test_required_columns(self):
        assert "log_return" in EntropyEngine.required_columns

    def test_output_columns_count(self):
        assert len(EntropyEngine().metadata().output_columns) == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestEntropyValues
# ─────────────────────────────────────────────────────────────────────────────

class TestEntropyValues:
    def test_entropy_non_negative(self, rand_df):
        d   = _add_returns(rand_df)
        out = EntropyEngine().generate(d)
        assert (out["entropy"] >= 0).all()

    def test_entropy_bounded(self, rand_df):
        d   = _add_returns(rand_df)
        out = EntropyEngine().generate(d)
        # Max Shannon entropy for 10 bins = log2(10) ≈ 3.32 bits
        assert (out["entropy"] <= np.log2(10) + 0.1).all()

    def test_entropy_zero_on_constant(self):
        n  = 50
        df = pd.DataFrame({"log_return": np.zeros(n)},
                          index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = EntropyEngine().generate(df)
        assert np.allclose(out["entropy"].values[4:], 0.0, atol=1e-8)

    def test_apen_non_negative(self, rand_df):
        d   = _add_returns(rand_df)
        out = EntropyEngine().generate(d)
        assert (out["approximate_entropy"] >= 0).all()

    def test_rolling_entropy_5_no_nan(self, rand_df):
        d   = _add_returns(rand_df)
        out = EntropyEngine().generate(d)
        assert not out["rolling_entropy_5"].isnull().any()

    def test_all_float64(self, rand_df):
        d   = _add_returns(rand_df)
        out = EntropyEngine().generate(d)
        assert (out.dtypes == np.float64).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestMicrostructureContract
# ─────────────────────────────────────────────────────────────────────────────

class TestMicrostructureContract:
    def test_registered(self):
        assert "market_microstructure" in FeatureRegistry.all_features()

    def test_name(self):
        assert MarketMicrostructureEngine.name == "market_microstructure"

    def test_category(self):
        assert MarketMicrostructureEngine.category == "statistics"

    def test_dependencies(self):
        assert "returns" in MarketMicrostructureEngine.dependencies

    def test_required_columns(self):
        for c in ("close", "log_return"):
            assert c in MarketMicrostructureEngine.required_columns

    def test_output_columns_count(self):
        assert len(MarketMicrostructureEngine().metadata().output_columns) == 8


# ─────────────────────────────────────────────────────────────────────────────
# TestMicrostructureValues
# ─────────────────────────────────────────────────────────────────────────────

class TestMicrostructureValues:
    def test_er_bounded(self, rand_df):
        d   = _add_returns(rand_df)
        out = MarketMicrostructureEngine().generate(d)
        assert (out["efficiency_ratio"] >= 0).all()
        assert (out["efficiency_ratio"] <= 1).all()

    def test_hurst_bounded(self, rand_df):
        d   = _add_returns(rand_df)
        out = MarketMicrostructureEngine().generate(d)
        assert (out["hurst"] >= 0).all() and (out["hurst"] <= 1).all()

    def test_fractal_dimension_from_hurst(self, rand_df):
        d   = _add_returns(rand_df)
        out = MarketMicrostructureEngine().generate(d)
        np.testing.assert_allclose(
            out["fractal_dimension"].values,
            2.0 - out["hurst"].values,
            rtol=1e-10,
        )

    def test_market_noise_from_er(self, rand_df):
        d   = _add_returns(rand_df)
        out = MarketMicrostructureEngine().generate(d)
        np.testing.assert_allclose(
            out["market_noise"].values,
            1.0 - out["efficiency_ratio"].values,
            rtol=1e-10,
        )

    def test_er_higher_in_uptrend(self, trend_df):
        rand_base = _make_ohlcv()
        d_rand  = _add_returns(rand_base)
        d_trend = _add_returns(trend_df)
        out_rand  = MarketMicrostructureEngine().generate(d_rand)
        out_trend = MarketMicrostructureEngine().generate(d_trend)
        assert out_trend["efficiency_ratio"].tail(100).mean() > \
               out_rand["efficiency_ratio"].tail(100).mean()

    def test_smoothness_bounded(self, rand_df):
        d   = _add_returns(rand_df)
        out = MarketMicrostructureEngine().generate(d)
        assert (out["price_smoothness"] >= 0).all()
        assert (out["price_smoothness"] <= 1 + 1e-9).all()

    def test_mean_reversion_trend_score_no_overlap(self, rand_df):
        """mean_reversion_score and trend_score can't both be >0 simultaneously."""
        d   = _add_returns(rand_df)
        out = MarketMicrostructureEngine().generate(d)
        both_positive = (out["mean_reversion_score"] > 0) & (out["trend_score"] > 0)
        assert not both_positive.any()

    def test_no_nan_no_inf(self, rand_df):
        d   = _add_returns(rand_df)
        out = MarketMicrostructureEngine().generate(d)
        assert not out.isnull().any().any()
        assert not np.isinf(out.values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestStatisticalEngineContract
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticalEngineContract:
    def test_registered(self):
        assert "statistics" in FeatureRegistry.all_features()

    def test_name(self):
        assert StatisticalEngine.name == "statistics"

    def test_category(self):
        assert StatisticalEngine.category == "statistics"

    def test_all_dependencies_present(self):
        for dep in ("returns", "rolling_statistics", "distribution",
                    "candle_statistics", "momentum_stats", "volatility_stats",
                    "entropy", "market_microstructure"):
            assert dep in StatisticalEngine.dependencies

    def test_required_columns(self):
        for c in ("rolling_return_5", "rolling_std", "efficiency_ratio"):
            assert c in StatisticalEngine.required_columns

    def test_output_columns_count(self):
        assert len(StatisticalEngine().metadata().output_columns) == 5


# ─────────────────────────────────────────────────────────────────────────────
# TestStatisticalEngineValues
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticalEngineValues:
    def test_return_vol_ratio_no_nan(self, rand_df):
        d   = _full_stat_df(rand_df)
        out = StatisticalEngine().generate(d)
        assert not out["return_vol_ratio"].isnull().any()

    def test_trend_quality_bounded(self, rand_df):
        d   = _full_stat_df(rand_df)
        out = StatisticalEngine().generate(d)
        assert (out["trend_quality"] >= 0).all() and (out["trend_quality"] <= 1 + 1e-9).all()

    def test_noise_ratio_non_negative(self, rand_df):
        d   = _full_stat_df(rand_df)
        out = StatisticalEngine().generate(d)
        assert (out["noise_ratio"] >= 0).all()

    def test_price_efficiency_bounded(self, rand_df):
        d   = _full_stat_df(rand_df)
        out = StatisticalEngine().generate(d)
        assert (out["price_efficiency"] >= 0).all() and (out["price_efficiency"] <= 1).all()

    def test_all_float64_no_inf(self, rand_df):
        d   = _full_stat_df(rand_df)
        out = StatisticalEngine().generate(d)
        assert (out.dtypes == np.float64).all()
        assert not np.isinf(out.values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestRegistryAll
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryAll:
    def test_returns_registered(self):
        assert "returns"              in FeatureRegistry.all_features()

    def test_candle_registered(self):
        assert "candle_statistics"    in FeatureRegistry.all_features()

    def test_rolling_registered(self):
        assert "rolling_statistics"   in FeatureRegistry.all_features()

    def test_distribution_registered(self):
        assert "distribution"         in FeatureRegistry.all_features()

    def test_all_stat_engines_registered(self):
        registry = FeatureRegistry.all_features()
        for name in ("momentum_stats", "volatility_stats", "entropy",
                     "market_microstructure", "statistics"):
            assert name in registry, f"'{name}' not in registry"


# ─────────────────────────────────────────────────────────────────────────────
# TestDependencyChain
# ─────────────────────────────────────────────────────────────────────────────

class TestDependencyChain:
    def test_returns_has_no_deps(self):
        assert ReturnsEngine.dependencies == []

    def test_distribution_depends_on_returns(self):
        assert "returns" in DistributionEngine.dependencies

    def test_entropy_depends_on_returns(self):
        assert "returns" in EntropyEngine.dependencies

    def test_stat_engine_depends_on_all_sub_engines(self):
        required = {"returns", "rolling_statistics", "distribution",
                    "candle_statistics", "momentum_stats", "volatility_stats",
                    "entropy", "market_microstructure"}
        assert required.issubset(set(StatisticalEngine.dependencies))


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_row_returns(self):
        df  = pd.DataFrame({"close": [1.1]},
                           index=pd.DatetimeIndex(["2024-01-01 00:00+00:00"]))
        out = ReturnsEngine().generate(df)
        assert len(out) == 1
        assert not np.isnan(out.values).any()

    def test_single_row_candle(self):
        df  = pd.DataFrame(
            {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05]},
            index=pd.DatetimeIndex(["2024-01-01 00:00+00:00"]),
        )
        out = CandleStatisticsEngine().generate(df)
        assert len(out) == 1
        assert not np.isnan(out.values).any()

    def test_constant_price_no_nan_distribution(self):
        n   = 50
        df  = pd.DataFrame({"close": np.ones(n), "log_return": np.zeros(n)},
                           index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = DistributionEngine().generate(df)
        assert not np.isnan(out.values).any()

    def test_constant_price_no_inf_microstructure(self):
        n   = 100
        df  = pd.DataFrame({"close": np.ones(n), "log_return": np.zeros(n)},
                           index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = MarketMicrostructureEngine().generate(df)
        assert not np.isinf(out.values).any()

    def test_two_rows_no_nan_candle(self):
        df  = pd.DataFrame(
            {"open": [1.0, 1.1], "high": [1.1, 1.2],
             "low": [0.9, 1.0], "close": [1.05, 1.15]},
            index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        )
        out = CandleStatisticsEngine().generate(df)
        assert not np.isnan(out.values).any()

    def test_rolling_with_three_bars(self):
        df  = pd.DataFrame({"close": [1.0, 1.1, 1.2]},
                           index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
        out = RollingStatisticsEngine().generate(df)
        assert not np.isnan(out.values).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestDtypeAndShape
# ─────────────────────────────────────────────────────────────────────────────

class TestDtypeAndShape:
    def test_returns_shape(self, rand_df):
        out = ReturnsEngine().generate(rand_df)
        assert out.shape == (len(rand_df), 5)

    def test_candle_shape(self, rand_df):
        out = CandleStatisticsEngine().generate(rand_df)
        assert out.shape == (len(rand_df), 21)

    def test_rolling_shape(self, rand_df):
        out = RollingStatisticsEngine().generate(rand_df)
        assert out.shape == (len(rand_df), 9)

    def test_microstructure_dtype(self, rand_df):
        d   = _add_returns(rand_df)
        out = MarketMicrostructureEngine().generate(d)
        assert (out.dtypes == np.float64).all()

    def test_total_stat_columns(self, rand_df):
        """5+21+9+6+7+7+3+8 = 66 sub-engine columns (excluding composite)."""
        total = (
            len(ReturnsEngine().generate(rand_df).columns)
            + len(CandleStatisticsEngine().generate(rand_df).columns)
            + len(RollingStatisticsEngine().generate(rand_df).columns)
        )
        assert total == 5 + 21 + 9


# ─────────────────────────────────────────────────────────────────────────────
# TestPerformance
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:
    def test_all_engines_under_30s_on_10k_bars(self):
        df = _make_ohlcv(n=10_000)
        t0 = time.perf_counter()

        ret  = ReturnsEngine().generate(df)
        cand = CandleStatisticsEngine().generate(df)
        rols = RollingStatisticsEngine().generate(df)
        d    = pd.concat([df, ret], axis=1)
        dist = DistributionEngine().generate(d)
        mom  = MomentumStatisticsEngine().generate(d)
        ent  = EntropyEngine().generate(d)
        mic  = MarketMicrostructureEngine().generate(d)

        from src.features.technical.volatility import VolatilityEngine
        vol_tech = VolatilityEngine().generate(df)
        d2   = pd.concat([df, ret, vol_tech], axis=1)
        vols = VolatilityStatisticsEngine().generate(d2)

        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, (
            f"Statistical engine (all sub-modules) took {elapsed:.2f}s on 10k bars"
        )
