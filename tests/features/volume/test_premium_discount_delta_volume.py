"""Tests for PremiumDiscountDeltaVolumeEngine.

Coverage:
  - Contract / registration (6 tests)
  - Output structure (5 tests)
  - Delta volume formula — Pine Script faithful (6 tests)
  - Zone-attributed volumes (6 tests)
  - Pressure / imbalance (5 tests)
  - Macro vs local delta (3 tests)
  - Volume regime signals (6 tests)
  - Exhaustion / expansion / compression (5 tests)
  - Edge cases (5 tests)
  - Integration / dtype (3 tests)
  - Performance (1 test)

Total: 51 tests
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.features.feature_registry import FeatureRegistry
from src.features.volume.premium_discount_delta_volume import (
    PremiumDiscountDeltaVolumeEngine,
    _OUTPUT_COLUMNS,
    _SR_PERIOD,
    _MACRO_PERIOD,
)

# ─── Fixtures / helpers ───────────────────────────────────────────────────────

_ENG = PremiumDiscountDeltaVolumeEngine()


def _make_df(
    n: int = 400,
    seed: int = 42,
    bull_frac: float = 0.5,
    vol_mean: float = 1_000.0,
) -> pd.DataFrame:
    """Synthetic OHLCV + pd_zone DataFrame."""
    rng = np.random.default_rng(seed)
    close = 1.0 + np.cumsum(rng.normal(0.0, 0.001, n))
    open_ = close + rng.normal(0.0, 0.0005, n)
    high  = np.maximum(close, open_) + np.abs(rng.normal(0.0, 0.0003, n))
    low   = np.minimum(close, open_) - np.abs(rng.normal(0.0, 0.0003, n))
    volume = np.abs(rng.normal(vol_mean, vol_mean * 0.3, n)) + 1.0

    # pd_zone: premium first third, equilibrium middle, discount last third
    pd_zone = np.zeros(n)
    pd_zone[: n // 3] = 1.0
    pd_zone[2 * n // 3 :] = -1.0

    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "pd_zone": pd_zone,
        },
        index=idx,
    )


def _make_all_bullish(n: int = 300, vol: float = 1_000.0) -> pd.DataFrame:
    """All bars close > open (purely bullish volume)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = np.linspace(1.0, 1.5, n)
    open_ = close - 0.001
    high  = close + 0.0005
    low   = open_ - 0.0005
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, vol),
            "pd_zone": np.zeros(n),
        },
        index=idx,
    )


def _make_all_bearish(n: int = 300, vol: float = 1_000.0) -> pd.DataFrame:
    """All bars close < open (purely bearish volume)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = np.linspace(1.5, 1.0, n)
    open_ = close + 0.001
    high  = open_ + 0.0005
    low   = close - 0.0005
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, vol),
            "pd_zone": np.zeros(n),
        },
        index=idx,
    )


def _make_alternating(n: int = 300, bull_vol: float = 100.0, bear_vol: float = 100.0) -> pd.DataFrame:
    """Alternating bullish / bearish bars with controllable volumes."""
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    is_bull = np.arange(n) % 2 == 0
    close = 1.0 + np.cumsum(np.where(is_bull, 0.001, -0.001))
    open_ = np.where(is_bull, close - 0.0005, close + 0.0005)
    high  = np.maximum(close, open_) + 0.0002
    low   = np.minimum(close, open_) - 0.0002
    volume = np.where(is_bull, bull_vol, bear_vol).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "pd_zone": np.zeros(n),
        },
        index=idx,
    )


def _make_large_df(n: int = 87_040) -> pd.DataFrame:
    return _make_df(n=n, seed=0)


# ─── Contract / registration ──────────────────────────────────────────────────

class TestContract:
    def test_engine_registered(self):
        assert "premium_discount_delta_volume" in FeatureRegistry.all_features()

    def test_engine_name(self):
        assert _ENG.name == "premium_discount_delta_volume"

    def test_engine_category(self):
        assert _ENG.category == "volume"

    def test_engine_dependencies(self):
        assert _ENG.dependencies == ["premium_discount"]

    def test_engine_required_columns(self):
        expected = {"open", "high", "low", "close", "volume", "pd_zone"}
        assert set(_ENG.required_columns) == expected

    def test_metadata_output_columns(self):
        meta = _ENG.metadata()
        assert meta.output_columns == _OUTPUT_COLUMNS


# ─── Output structure ─────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_output_has_all_20_columns(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert list(out.columns) == _OUTPUT_COLUMNS

    def test_output_exactly_20_columns(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert len(out.columns) == 20

    def test_output_index_matches_input(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert out.index.equals(df.index)

    def test_output_row_count_matches_input(self):
        df = _make_df(n=200)
        out = _ENG.generate(df)
        assert len(out) == len(df)

    def test_validate_output_passes(self):
        df = _make_df()
        out = _ENG.generate(df)
        _ENG.validate_output(df, out)  # must not raise


# ─── Delta volume formula (Pine Script faithful) ──────────────────────────────

class TestDeltaVolumeFormula:
    def test_all_bullish_delta_is_100(self):
        """Pure buying: neg_avg = 0, so (0/pos + 1)*100 = 100."""
        df = _make_all_bullish(n=300)
        out = _ENG.generate(df)
        # After warm-up (first SR_PERIOD bars) all values should be exactly 100
        tail = out["delta_volume"].iloc[_SR_PERIOD:]
        assert (tail == 100.0).all(), f"Expected 100.0, got {tail.unique()}"

    def test_all_bearish_delta_is_neg100(self):
        """Pure selling: pos_avg = 0, so clipped to -100."""
        df = _make_all_bearish(n=300)
        out = _ENG.generate(df)
        tail = out["delta_volume"].iloc[_SR_PERIOD:]
        assert (tail == -100.0).all(), f"Expected -100.0, got {tail.unique()}"

    def test_balanced_volume_delta_near_zero(self):
        """Equal bull/bear volumes → delta ≈ 0."""
        df = _make_alternating(n=300, bull_vol=100.0, bear_vol=100.0)
        out = _ENG.generate(df)
        tail = out["delta_volume"].iloc[_SR_PERIOD:]
        assert np.allclose(tail.to_numpy(), 0.0, atol=2.0), (
            f"Expected ~0, max abs = {np.abs(tail.to_numpy()).max():.3f}"
        )

    def test_heavier_bear_volume_gives_negative_delta(self):
        """Bears have 3× more volume → delta must be negative."""
        df = _make_alternating(n=300, bull_vol=100.0, bear_vol=300.0)
        out = _ENG.generate(df)
        tail = out["delta_volume"].iloc[_SR_PERIOD:]
        assert (tail < 0).all()

    def test_delta_clamped_upper(self):
        df = _make_all_bullish(n=300)
        out = _ENG.generate(df)
        assert (out["delta_volume"] <= 100.0).all()

    def test_delta_clamped_lower(self):
        df = _make_all_bearish(n=300)
        out = _ENG.generate(df)
        assert (out["delta_volume"] >= -100.0).all()

    def test_delta_percent_is_delta_over_100(self):
        df = _make_df()
        out = _ENG.generate(df)
        np.testing.assert_array_almost_equal(
            out["delta_percent"].to_numpy(),
            out["delta_volume"].to_numpy() / 100.0,
        )

    def test_local_delta_equals_delta_volume(self):
        df = _make_df()
        out = _ENG.generate(df)
        np.testing.assert_array_equal(
            out["local_delta"].to_numpy(), out["delta_volume"].to_numpy()
        )


# ─── Zone-attributed volumes ──────────────────────────────────────────────────

class TestZoneVolumes:
    def test_premium_volume_zero_when_no_premium_bars(self):
        df = _make_df(n=300)
        df["pd_zone"] = -1.0  # all discount
        out = _ENG.generate(df)
        assert (out["premium_volume"] == 0.0).all()

    def test_discount_volume_zero_when_no_discount_bars(self):
        df = _make_df(n=300)
        df["pd_zone"] = 1.0  # all premium
        out = _ENG.generate(df)
        assert (out["discount_volume"] == 0.0).all()

    def test_equilibrium_volume_zero_when_no_eq_bars(self):
        df = _make_df(n=300)
        df["pd_zone"] = 1.0
        out = _ENG.generate(df)
        assert (out["equilibrium_volume"] == 0.0).all()

    def test_premium_volume_matches_bar_volumes_in_premium_zone(self):
        """In the first 100 bars (all premium), positive_volume == premium rolling mean."""
        df = _make_df(n=300)
        df["pd_zone"] = 1.0  # single zone
        out = _ENG.generate(df)
        # When all bars are premium, premium_volume must equal the rolling mean of volume
        expected = pd.Series(df["volume"].to_numpy()).rolling(_SR_PERIOD, min_periods=1).mean()
        np.testing.assert_array_almost_equal(
            out["premium_volume"].to_numpy(), expected.to_numpy(), decimal=6
        )

    def test_zone_volumes_nonnegative(self):
        df = _make_df()
        out = _ENG.generate(df)
        for col in ("premium_volume", "discount_volume", "equilibrium_volume"):
            assert (out[col] >= 0.0).all(), f"{col} has negative values"

    def test_premium_strength_range(self):
        """premium_strength in [0, 100]."""
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["premium_strength"] >= 0.0).all()
        assert (out["premium_strength"] <= 100.0).all()

    def test_discount_strength_range(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["discount_strength"] >= 0.0).all()
        assert (out["discount_strength"] <= 100.0).all()

    def test_premium_strength_high_when_all_premium_bars_are_bearish(self):
        """When premium zone has only bearish bars, sell dominates → high strength."""
        df = _make_all_bearish(n=300)
        df["pd_zone"] = 1.0
        out = _ENG.generate(df)
        tail = out["premium_strength"].iloc[_SR_PERIOD:]
        assert (tail == 100.0).all()

    def test_discount_strength_high_when_all_discount_bars_are_bullish(self):
        df = _make_all_bullish(n=300)
        df["pd_zone"] = -1.0
        out = _ENG.generate(df)
        tail = out["discount_strength"].iloc[_SR_PERIOD:]
        assert (tail == 100.0).all()


# ─── Pressure / imbalance ─────────────────────────────────────────────────────

class TestPressureAndImbalance:
    def test_buy_pressure_range(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["buy_pressure"] >= 0.0).all()
        assert (out["buy_pressure"] <= 1.0).all()

    def test_sell_pressure_range(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["sell_pressure"] >= 0.0).all()
        assert (out["sell_pressure"] <= 1.0).all()

    def test_buy_plus_sell_le_one(self):
        """buy_pressure + sell_pressure ≤ 1.0 (neutral / doji bars are excluded)."""
        df = _make_df()
        out = _ENG.generate(df)
        total = out["buy_pressure"] + out["sell_pressure"]
        assert (total <= 1.0 + 1e-9).all(), f"max={total.max():.6f}"

    def test_volume_imbalance_range(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["volume_imbalance"] >= -1.0).all()
        assert (out["volume_imbalance"] <= 1.0).all()

    def test_volume_imbalance_positive_when_buy_dominant(self):
        """All-bullish → imbalance should be +1."""
        df = _make_all_bullish(n=300)
        out = _ENG.generate(df)
        tail = out["volume_imbalance"].iloc[_SR_PERIOD:]
        assert (tail == 1.0).all()

    def test_volume_imbalance_negative_when_sell_dominant(self):
        df = _make_all_bearish(n=300)
        out = _ENG.generate(df)
        tail = out["volume_imbalance"].iloc[_SR_PERIOD:]
        assert (tail == -1.0).all()


# ─── Macro vs local delta ─────────────────────────────────────────────────────

class TestMacroVsLocalDelta:
    def test_macro_delta_range(self):
        df = _make_df(n=500)
        out = _ENG.generate(df)
        assert (out["macro_delta"] >= -100.0).all()
        assert (out["macro_delta"] <= 100.0).all()

    def test_local_delta_range(self):
        df = _make_df(n=500)
        out = _ENG.generate(df)
        assert (out["delta_volume"] >= -100.0).all()
        assert (out["delta_volume"] <= 100.0).all()

    def test_macro_delta_smoother_than_local(self):
        """Macro delta (longer window) should have lower std dev than local delta."""
        df = _make_df(n=1_000, seed=7)
        out = _ENG.generate(df)
        tail = out.iloc[_MACRO_PERIOD:]
        assert tail["macro_delta"].std() <= tail["delta_volume"].std() * 1.05, (
            "macro_delta is not smoother than delta_volume"
        )


# ─── Volume regime signals ────────────────────────────────────────────────────

class TestVolumeRegimeSignals:
    def test_volume_strength_positive(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["volume_strength"] > 0.0).all()

    def test_volume_strength_near_one_on_constant_volume(self):
        """Constant volume → strength ≈ 1.0 after warm-up."""
        df = _make_df(n=500)
        df["volume"] = 1_000.0
        out = _ENG.generate(df)
        tail = out["volume_strength"].iloc[_MACRO_PERIOD:]
        assert np.allclose(tail.to_numpy(), 1.0, atol=0.01)

    def test_volume_trend_values_in_neg1_0_pos1(self):
        df = _make_df()
        out = _ENG.generate(df)
        unique = set(out["volume_trend"].round(6).unique())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_volume_acceleration_finite(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert np.isfinite(out["volume_acceleration"].to_numpy()).all()

    def test_positive_volume_nonneg(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["positive_volume"] >= 0.0).all()

    def test_negative_volume_nonneg(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["negative_volume"] >= 0.0).all()


# ─── Exhaustion / expansion / compression ─────────────────────────────────────

class TestExhaustionExpansionCompression:
    def test_volume_exhaustion_nonneg(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["volume_exhaustion"] >= 0.0).all()

    def test_volume_expansion_nonneg(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["volume_expansion"] >= 0.0).all()

    def test_volume_compression_nonneg(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert (out["volume_compression"] >= 0.0).all()

    def test_expansion_zero_when_volume_below_average(self):
        """Low-volume bars (half the mean) should give expansion = 0."""
        df = _make_df(n=400)
        # Set last 50 bars to very low volume so expansion = 0
        v = df["volume"].to_numpy().copy()
        mean_vol = v[:300].mean()
        v[350:] = mean_vol * 0.1
        df["volume"] = v
        out = _ENG.generate(df)
        tail_exp = out["volume_expansion"].iloc[350:]
        assert (tail_exp == 0.0).all()

    def test_compression_zero_when_volume_above_average(self):
        """Very high volume bars should give compression ≈ 0 (floating-point tolerance)."""
        df = _make_df(n=400)
        v = df["volume"].to_numpy().copy()
        mean_vol = v[:300].mean()
        v[350:] = mean_vol * 10.0
        df["volume"] = v
        out = _ENG.generate(df)
        tail_comp = out["volume_compression"].iloc[350:]
        assert (tail_comp < 1e-9).all()


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_doji_bars_gives_neutral_delta(self):
        """Bars where close == open contribute 0 to both sides → delta = 0."""
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="15min")
        price = np.ones(n) * 1.0
        df = pd.DataFrame(
            {
                "open":    price,
                "high":    price + 0.001,
                "low":     price - 0.001,
                "close":   price,
                "volume":  np.ones(n) * 1_000.0,
                "pd_zone": np.zeros(n),
            },
            index=idx,
        )
        out = _ENG.generate(df)
        # Both pos_avg and neg_avg are 0 → safe_delta returns 0
        assert (out["delta_volume"] == 0.0).all()

    def test_nan_pd_zone_treated_as_equilibrium(self):
        """NaN in pd_zone must not propagate to output."""
        df = _make_df(n=300)
        df.loc[df.index[:50], "pd_zone"] = np.nan
        out = _ENG.generate(df)
        assert np.isfinite(out.to_numpy()).all()

    def test_single_bar_does_not_crash(self):
        df = _make_df(n=1)
        out = _ENG.generate(df)
        assert len(out) == 1

    def test_minimum_data_two_bars(self):
        df = _make_df(n=2)
        out = _ENG.generate(df)
        assert len(out) == 2

    def test_constant_volume_returns_finite_output(self):
        df = _make_df(n=300)
        df["volume"] = 500.0
        out = _ENG.generate(df)
        assert np.isfinite(out.to_numpy()).all()


# ─── Integration / dtype ──────────────────────────────────────────────────────

class TestIntegrationAndDtype:
    def test_output_is_dataframe(self):
        df = _make_df()
        out = _ENG.generate(df)
        assert isinstance(out, pd.DataFrame)

    def test_all_columns_float64(self):
        df = _make_df()
        out = _ENG.generate(df)
        for col in out.columns:
            assert out[col].dtype == np.float64, f"{col} is {out[col].dtype}"

    def test_no_nans_in_output(self):
        df = _make_df(n=400)
        out = _ENG.generate(df)
        nan_counts = out.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
        assert not cols_with_nan, f"NaN columns: {cols_with_nan}"


# ─── Performance ──────────────────────────────────────────────────────────────

class TestPerformance:
    def test_87k_rows_under_5s(self):
        df = _make_large_df(n=87_040)
        t0 = time.perf_counter()
        out = _ENG.generate(df)
        elapsed = time.perf_counter() - t0
        assert len(out) == 87_040
        assert elapsed < 5.0, f"generate() took {elapsed:.2f}s on 87K rows"
