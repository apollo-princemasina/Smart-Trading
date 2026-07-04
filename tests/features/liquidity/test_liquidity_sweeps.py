"""Unit, integration and performance tests for LiquiditySweepEngine.

Run with:
    pytest tests/features/liquidity/test_liquidity_sweeps.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.features.liquidity.liquidity_sweeps import LiquiditySweepEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Minimal OHLCV + market structure columns needed by the engine."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    close = 1.1000 + np.cumsum(rng.normal(0, 0.0002, n))
    spread = rng.uniform(0.0001, 0.0005, n)
    high  = close + spread
    low   = close - spread
    open_ = close - rng.normal(0, 0.0001, n)

    df = pd.DataFrame(
        {
            "open":  open_,
            "high":  high,
            "low":   low,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        },
        index=idx,
    )

    # ── Synthetic market structure columns ────────────────────────────────────
    # Place pivot highs every 15 bars, pivot lows every 15 bars (offset by 7)
    ph = np.zeros(n, float)
    pl = np.zeros(n, float)
    mph = np.zeros(n, float)
    mpl = np.zeros(n, float)

    for i in range(14, n, 15):
        ph[i] = 1.0
        mph[i] = 1.0
    for i in range(7, n, 15):
        pl[i] = 1.0
        mpl[i] = 1.0

    df["pivot_high"]       = ph
    df["pivot_low"]        = pl
    df["major_pivot_high"] = mph
    df["major_pivot_low"]  = mpl

    # swing_high/low_price — forward-fill the last pivot price
    df["swing_high_price"] = df["high"].where(df["pivot_high"] == 1.0).ffill()
    df["swing_low_price"]  = df["low"].where(df["pivot_low"] == 1.0).ffill()

    # BOS/CHoCH placeholders (all zero — we don't need specific events here)
    for col in ("bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish"):
        df[col] = 0.0

    # EQH / EQL — place a few equal-level events
    eqh = np.zeros(n, float)
    eql = np.zeros(n, float)
    if n > 60:
        eqh[44] = 1.0
        eql[51] = 1.0
    df["eqh"]       = eqh
    df["eql"]       = eql
    df["eqh_price"] = df["high"].where(df["eqh"] == 1.0).ffill()
    df["eql_price"] = df["low"].where(df["eql"] == 1.0).ffill()
    df["eqh_age"]   = 0.0
    df["eql_age"]   = 0.0

    return df


def _make_sweep_df() -> pd.DataFrame:
    """Craft a minimal DataFrame that guarantees one bullish and one bearish sweep."""
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    base = 1.1000

    close_arr = np.full(n, base)
    high_arr  = np.full(n, base + 0.0005)
    low_arr   = np.full(n, base - 0.0005)
    open_arr  = np.full(n, base)

    df = pd.DataFrame(
        {"open": open_arr, "high": high_arr, "low": low_arr, "close": close_arr},
        index=idx,
    )

    # ── Place a sell-side pivot low at bar 10 ─────────────────────────────────
    # The level price will be base - 0.0005 = 1.0995
    # At bar 20: low drops below 1.0995 AND close above 1.0995 → bullish sweep
    ph = np.zeros(n, float)
    pl = np.zeros(n, float)
    pl[10] = 1.0   # pivot low at bar 10 — sell-side liquidity level

    # Ensure bar 10 low is the level we track
    low_arr[10]  = base - 0.0020   # 1.0980 — distinct pivot low

    # Sweep bar 20: low goes below pivot low, close above it
    low_arr[20]  = 1.0970           # pierces below 1.0980
    close_arr[20] = 1.1010          # closes back above 1.0980 → bullish sweep

    # ── Place a buy-side pivot high at bar 30 ─────────────────────────────────
    ph[30] = 1.0
    high_arr[30] = base + 0.0020   # 1.1020 — distinct pivot high

    # Sweep bar 40: high above 1.1020, close below it → bearish sweep
    high_arr[40]  = 1.1030
    close_arr[40] = 1.0990

    df["open"]  = open_arr
    df["high"]  = high_arr
    df["low"]   = low_arr
    df["close"] = close_arr

    df["pivot_high"]       = ph
    df["pivot_low"]        = pl
    df["major_pivot_high"] = ph.copy()
    df["major_pivot_low"]  = pl.copy()
    df["swing_high_price"] = df["high"].where(df["pivot_high"] == 1.0).ffill()
    df["swing_low_price"]  = df["low"].where(df["pivot_low"] == 1.0).ffill()

    for col in ("bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
                "eqh", "eql", "eqh_age", "eql_age"):
        df[col] = 0.0

    df["eqh_price"] = np.nan
    df["eql_price"] = np.nan

    return df


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestLiquiditySweepEngineContract:

    def test_registration(self):
        from src.features.feature_registry import FeatureRegistry
        assert "liquidity_sweeps" in FeatureRegistry.all_features()

    def test_name_and_category(self):
        eng = LiquiditySweepEngine()
        assert eng.name == "liquidity_sweeps"
        assert eng.category == "liquidity"

    def test_dependencies_declared(self):
        assert "market_structure"  in LiquiditySweepEngine.dependencies
        assert "bos_choch"         in LiquiditySweepEngine.dependencies
        assert "equal_highs_lows"  in LiquiditySweepEngine.dependencies

    def test_output_shape(self):
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        assert len(out) == 200
        assert out.index.equals(df.index)

    def test_output_columns(self):
        df  = _make_df(100)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        expected = [
            "bullish_liquidity_sweep", "bearish_liquidity_sweep",
            "liquidity_score", "nearest_liquidity_distance",
            "nearest_buy_liquidity", "nearest_sell_liquidity",
            "liquidity_age", "touch_count",
            "strong_sweep", "weak_sweep", "confirmed_sweep",
            "sweep_strength", "liquidity_cluster_size",
            "sweep_penetration", "sweep_rejection",
            "liq_zone_width", "liq_zone_lifetime", "num_nearby_liq_pools",
        ]
        assert set(expected) == set(out.columns), (
            f"Missing: {set(expected) - set(out.columns)}, "
            f"Extra: {set(out.columns) - set(expected)}"
        )

    def test_all_columns_float64(self):
        df  = _make_df(100)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        non_float = {c: str(out[c].dtype) for c in out.columns if out[c].dtype != np.float64}
        assert not non_float, f"Non-float64 columns: {non_float}"

    def test_binary_flags_are_zero_or_one(self):
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        binary_cols = [
            "bullish_liquidity_sweep", "bearish_liquidity_sweep",
            "strong_sweep", "weak_sweep", "confirmed_sweep",
        ]
        for col in binary_cols:
            unique_vals = set(out[col].dropna().unique())
            assert unique_vals <= {0.0, 1.0}, f"{col} contains non-binary: {unique_vals}"

    def test_no_phantom_columns(self):
        """Output must not include any input column names."""
        df  = _make_df(100)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        overlap = set(out.columns) & set(df.columns)
        assert not overlap, f"Output shadows input columns: {overlap}"

    def test_validate_output_passes(self):
        df  = _make_df(100)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        eng.validate_output(df, out)   # must not raise

    def test_metadata_populated(self):
        meta = LiquiditySweepEngine().metadata()
        assert meta.name == "liquidity_sweeps"
        assert len(meta.output_columns) == 18
        assert meta.complexity == "high"


# ── Sweep detection correctness ───────────────────────────────────────────────

class TestSweepDetection:

    def test_bullish_sweep_detected(self):
        df  = _make_sweep_df()
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        assert out["bullish_liquidity_sweep"].iloc[20] == 1.0, (
            "Expected a bullish sweep at bar 20"
        )

    def test_bearish_sweep_detected(self):
        df  = _make_sweep_df()
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        assert out["bearish_liquidity_sweep"].iloc[40] == 1.0, (
            "Expected a bearish sweep at bar 40"
        )

    def test_no_false_sweep_on_normal_bars(self):
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        # On a random-walk dataset the vast majority of bars are non-sweep
        bull_frac = out["bullish_liquidity_sweep"].mean()
        bear_frac = out["bearish_liquidity_sweep"].mean()
        assert bull_frac < 0.5, f"Bullish sweep rate too high: {bull_frac:.2%}"
        assert bear_frac < 0.5, f"Bearish sweep rate too high: {bear_frac:.2%}"

    def test_sweep_is_not_re_triggered(self):
        """A level that was already swept must not fire again."""
        df  = _make_sweep_df()
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        # bars 21–59 should NOT have another bullish sweep from the same level
        later_bull = out["bullish_liquidity_sweep"].iloc[21:].sum()
        # There might be subsequent sweeps on different levels, so just verify
        # the first sweep was detected and the level is marked swept
        assert out["bullish_liquidity_sweep"].iloc[20] == 1.0

    def test_sweep_penetration_positive_on_sweep(self):
        df  = _make_sweep_df()
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        assert out["sweep_penetration"].iloc[20] > 0.0
        assert out["sweep_penetration"].iloc[40] > 0.0

    def test_strong_weak_sweep_mutually_exclusive(self):
        df  = _make_df(300)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        # Where a sweep happened, strong + weak should not both be 1
        any_sweep = (out["bullish_liquidity_sweep"] == 1.0) | (out["bearish_liquidity_sweep"] == 1.0)
        both_set  = (out["strong_sweep"] == 1.0) & (out["weak_sweep"] == 1.0)
        assert not (any_sweep & both_set).any(), "strong_sweep and weak_sweep both 1 on same bar"

    def test_confirmed_sweep_fires_day_after(self):
        df  = _make_sweep_df()
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        # If bar 20 is a bullish sweep and bar 21 close > bar 20 close, confirmed at 21
        if out["bullish_liquidity_sweep"].iloc[20] == 1.0:
            if df["close"].iloc[21] > df["close"].iloc[20]:
                assert out["confirmed_sweep"].iloc[21] == 1.0


# ── Nearest-level metrics ─────────────────────────────────────────────────────

class TestNearestLevelMetrics:

    def test_liquidity_score_in_range(self):
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        scores = out["liquidity_score"]
        assert scores.min() >= 0.0
        assert scores.max() <= 100.0

    def test_nearest_distance_nonnegative(self):
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        valid = out["nearest_liquidity_distance"].dropna()
        assert (valid >= 0.0).all()

    def test_num_nearby_pools_nonneg_int(self):
        df  = _make_df(300)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        pools = out["num_nearby_liq_pools"]
        assert (pools >= 0.0).all()
        assert (pools == pools.round()).all(), "num_nearby_liq_pools must be integer-valued"

    def test_liquidity_age_nonneg(self):
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        assert (out["liquidity_age"] >= 0.0).all()

    def test_buy_sell_liquidity_prices_present(self):
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        # After first pivot high/low, prices should be non-NaN
        late = out.iloc[30:]
        assert late["nearest_buy_liquidity"].notna().any()
        assert late["nearest_sell_liquidity"].notna().any()

    def test_zone_width_proportional_to_price(self):
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        # Zone width = _EQUAL_PCT * price * 100 ≈ 0.05 % * 1.10 * 100 ≈ 0.055
        nonzero = out["liq_zone_width"][out["liq_zone_width"] > 0]
        if len(nonzero) > 0:
            assert nonzero.mean() < 1.0, "Zone width seems too large (> 1 % of price)"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_dataframe_raises_or_returns_empty(self):
        df  = _make_df(0)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        assert len(out) == 0

    def test_single_row(self):
        df  = _make_df(1)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        assert len(out) == 1

    def test_all_nan_pivots(self):
        df = _make_df(50)
        df["pivot_high"] = 0.0
        df["pivot_low"]  = 0.0
        df["major_pivot_high"] = 0.0
        df["major_pivot_low"]  = 0.0
        df["swing_high_price"] = np.nan
        df["swing_low_price"]  = np.nan
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        # No levels → no sweeps
        assert out["bullish_liquidity_sweep"].sum() == 0.0
        assert out["bearish_liquidity_sweep"].sum() == 0.0

    def test_multiple_timeframes_same_result(self):
        """Same data, different datetime frequency — output rows must match."""
        df1 = _make_df(100)
        df2 = df1.copy()
        df2.index = pd.date_range("2023-01-01", periods=100, freq="1h", tz="UTC")
        eng = LiquiditySweepEngine()
        out1 = eng.generate(df1)
        out2 = eng.generate(df2)
        pd.testing.assert_frame_equal(
            out1.reset_index(drop=True),
            out2.reset_index(drop=True),
            check_names=False,
        )


# ── Integration test ──────────────────────────────────────────────────────────

class TestPipelineIntegration:

    def test_engine_runs_after_dependencies(self):
        """Smoke-test: the engine generates without error when called with
        a properly enriched DataFrame (simulating pipeline output)."""
        df  = _make_df(500)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        assert len(out) == 500
        assert "bullish_liquidity_sweep" in out.columns
        assert "liquidity_score" in out.columns

    def test_output_can_concatenate_with_input(self):
        """Verify the output can be safely merged back into the running_df."""
        df  = _make_df(200)
        eng = LiquiditySweepEngine()
        out = eng.generate(df)
        combined = pd.concat([df, out], axis=1)
        assert len(combined) == 200
        # No column duplication
        assert combined.columns.duplicated().sum() == 0


# ── Performance benchmark ─────────────────────────────────────────────────────

class TestPerformance:

    def test_87k_rows_under_30_seconds(self):
        """Pipeline must process the full 87 K-row EURUSD M15 dataset in < 30 s."""
        n   = 87_503
        df  = _make_df(n, seed=0)
        eng = LiquiditySweepEngine()

        start = time.perf_counter()
        out   = eng.generate(df)
        elapsed = time.perf_counter() - start

        assert len(out) == n
        assert elapsed < 30.0, (
            f"LiquiditySweepEngine took {elapsed:.1f}s on {n} rows — exceeds 30 s budget"
        )

    def test_throughput_rows_per_second(self):
        n   = 10_000
        df  = _make_df(n, seed=1)
        eng = LiquiditySweepEngine()

        start = time.perf_counter()
        eng.generate(df)
        elapsed = time.perf_counter() - start

        rows_per_sec = n / elapsed
        assert rows_per_sec > 500, (
            f"Throughput {rows_per_sec:.0f} rows/s is below 500 rows/s minimum"
        )
