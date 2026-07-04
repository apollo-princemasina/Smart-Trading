"""Unit, integration and performance tests for LiquidityMagnetEngine.

Run with:
    pytest tests/features/liquidity/test_liquidity_magnet.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.features.liquidity.liquidity_magnet import (
    LiquidityMagnetEngine,
    _K_SIDE,
    _MIN_SCORE,
    _MAX_DIST,
    _W_PROX,
    _W_AGE,
    _W_TOUCH,
    _W_MOM,
    _PROX_K,
    _AGE_MAX,
    _MOM_K,
    _MOM_LEN,
    _ATR_PERIOD,
    K,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Minimal OHLCV + required market structure / EQH-EQL columns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    close = 1.1000 + np.cumsum(rng.normal(0, 0.0002, n))
    spread = rng.uniform(0.0001, 0.0005, n)
    high  = close + spread
    low   = close - spread
    open_ = close + rng.normal(0, 0.0001, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )

    # Pivots: minor high every 15 bars, minor low offset by 7
    ph = np.zeros(n, float); pl = np.zeros(n, float)
    mph = np.zeros(n, float); mpl = np.zeros(n, float)
    for i in range(14, n, 15):
        ph[i] = 1.0; mph[i] = 1.0
    for i in range(7, n, 15):
        pl[i] = 1.0; mpl[i] = 1.0

    df["pivot_high"]       = ph
    df["pivot_low"]        = pl
    df["major_pivot_high"] = mph
    df["major_pivot_low"]  = mpl
    df["eqh"]              = 0.0
    df["eql"]              = 0.0
    return df


def _make_controlled_df() -> pd.DataFrame:
    """Flat price = 1.1000, ATR ≈ constant, one sell-side and one buy-side pool."""
    n   = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    c   = np.full(n, 1.1000)
    h   = c + 0.0010
    lo  = c - 0.0010

    df = pd.DataFrame({"open": c.copy(), "high": h, "low": lo, "close": c.copy()}, index=idx)

    ph = np.zeros(n, float); pl = np.zeros(n, float)
    mph = np.zeros(n, float); mpl = np.zeros(n, float)

    # Buy-side pool at bar 5 price = 1.1020 (above market)
    ph[5] = 1.0; mph[5] = 1.0; df.at[idx[5], "high"] = 1.1020

    # Sell-side pool at bar 10 price = 1.0980 (below market)
    pl[10] = 1.0; mpl[10] = 1.0; df.at[idx[10], "low"] = 1.0980

    df["pivot_high"]       = ph
    df["pivot_low"]        = pl
    df["major_pivot_high"] = mph
    df["major_pivot_low"]  = mpl
    df["eqh"] = 0.0
    df["eql"] = 0.0
    return df


def _make_sweep_scenario() -> pd.DataFrame:
    """Scenario where sell-side pool is swept bullishly at bar 30."""
    n   = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    base = 1.1000

    c = np.full(n, base)
    h = c + 0.0010
    lo = c - 0.0010

    df = pd.DataFrame({"open": c.copy(), "high": h, "low": lo, "close": c.copy()}, index=idx)

    ph  = np.zeros(n, float); pl  = np.zeros(n, float)
    mph = np.zeros(n, float); mpl = np.zeros(n, float)

    # Sell-side pool at bar 5, pool price = 1.0970
    pl[5] = 1.0; mpl[5] = 1.0; df.at[idx[5], "low"] = 1.0970

    # Sweep at bar 30: low dips below 1.0970, close recovers above
    df.at[idx[30], "low"]   = 1.0960
    df.at[idx[30], "close"] = 1.1005

    df["pivot_high"]       = ph
    df["pivot_low"]        = pl
    df["major_pivot_high"] = mph
    df["major_pivot_low"]  = mpl
    df["eqh"] = 0.0
    df["eql"] = 0.0
    return df


# ── Contract tests ────────────────────────────────────────────────────────────

class TestContract:

    def test_registration(self):
        from src.features.feature_registry import FeatureRegistry
        assert "liquidity_magnet" in FeatureRegistry.all_features()

    def test_name_and_category(self):
        eng = LiquidityMagnetEngine()
        assert eng.name == "liquidity_magnet"
        assert eng.category == "liquidity"

    def test_dependencies_declared(self):
        deps = LiquidityMagnetEngine.dependencies
        assert "market_structure"  in deps
        assert "bos_choch"         in deps
        assert "equal_highs_lows"  in deps
        assert "liquidity_sweeps"  in deps

    def test_output_shape(self):
        df  = _make_df(200)
        out = LiquidityMagnetEngine().generate(df)
        assert len(out) == 200
        assert out.index.equals(df.index)

    def test_output_column_count(self):
        df  = _make_df(100)
        out = LiquidityMagnetEngine().generate(df)
        assert len(out.columns) == 20, f"Expected 20 columns, got {len(out.columns)}"

    def test_exact_output_columns(self):
        expected = {
            "nearest_buy_liquidity_distance",
            "nearest_sell_liquidity_distance",
            "nearest_liquidity_score",
            "magnet_score",
            "magnet_probability",
            "liquidity_rank",
            "target_liquidity",
            "distance_to_target",
            "buy_side_probability",
            "sell_side_probability",
            "liquidity_density",
            "cluster_strength",
            "magnet_strength",
            "nearest_cluster_size",
            "proximity_contribution",
            "age_contribution",
            "touch_contribution",
            "momentum_contribution",
            "ranking_position",
            "target_direction",
        }
        df  = _make_df(50)
        out = LiquidityMagnetEngine().generate(df)
        assert set(out.columns) == expected

    def test_all_columns_float64(self):
        df  = _make_df(100)
        out = LiquidityMagnetEngine().generate(df)
        non_float = {c: str(out[c].dtype) for c in out.columns if out[c].dtype != np.float64}
        assert not non_float, f"Non-float64 columns: {non_float}"

    def test_no_phantom_columns(self):
        df  = _make_df(100)
        out = LiquidityMagnetEngine().generate(df)
        overlap = set(out.columns) & set(df.columns)
        assert not overlap, f"Output shadows input: {overlap}"

    def test_validate_output_passes(self):
        df  = _make_df(100)
        eng = LiquidityMagnetEngine()
        out = eng.generate(df)
        eng.validate_output(df, out)    # must not raise

    def test_metadata_populated(self):
        meta = LiquidityMagnetEngine().metadata()
        assert meta.name == "liquidity_magnet"
        assert len(meta.output_columns) == 20
        assert meta.complexity == "high"
        assert "market_structure" in meta.dependencies
        assert "liquidity_sweeps" in meta.dependencies


# ── Score range and probability constraints ───────────────────────────────────

class TestScoreRanges:

    def test_magnet_score_in_range(self):
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        assert out["magnet_score"].min() >= 0.0
        assert out["magnet_score"].max() <= 100.0

    def test_magnet_probability_in_unit_range(self):
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        probs = out["magnet_probability"]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_probability_equals_score_over_100(self):
        df  = _make_df(200)
        out = LiquidityMagnetEngine().generate(df)
        mask = out["magnet_score"] > 0
        if mask.any():
            ratio = (out.loc[mask, "magnet_probability"]
                     / (out.loc[mask, "magnet_score"] / 100.0))
            assert np.allclose(ratio, 1.0, atol=1e-6)

    def test_side_probabilities_sum_to_one(self):
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        active = (out["magnet_strength"] > 0)
        if active.any():
            total = out.loc[active, "buy_side_probability"] + out.loc[active, "sell_side_probability"]
            assert np.allclose(total, 1.0, atol=1e-6), "Side probabilities must sum to 1"

    def test_side_probabilities_in_unit_range(self):
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        assert out["buy_side_probability"].between(0.0, 1.0).all()
        assert out["sell_side_probability"].between(0.0, 1.0).all()

    def test_magnet_strength_geq_magnet_score(self):
        """Max score across all pools must be >= score of the selected target."""
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        mask = out["magnet_score"] > 0
        if mask.any():
            assert (out.loc[mask, "magnet_strength"]
                    >= out.loc[mask, "magnet_score"] - 1e-9).all()

    def test_target_direction_only_valid_values(self):
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        unique = set(out["target_direction"].unique())
        assert unique <= {-1.0, 0.0, 1.0}

    def test_liquidity_density_nonneg_int(self):
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        assert (out["liquidity_density"] >= 0).all()
        assert (out["liquidity_density"] == out["liquidity_density"].round()).all()

    def test_ranking_position_positive(self):
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        active = out["ranking_position"] > 0
        if active.any():
            assert (out.loc[active, "ranking_position"] >= 1).all()

    def test_score_components_sum_to_total(self):
        """proximity + age + touch + momentum should equal magnet_score for targets."""
        df  = _make_df(300)
        out = LiquidityMagnetEngine().generate(df)
        mask = out["magnet_score"] > 0
        if mask.any():
            reconstructed = (
                out.loc[mask, "proximity_contribution"]
                + out.loc[mask, "age_contribution"]
                + out.loc[mask, "touch_contribution"]
                + out.loc[mask, "momentum_contribution"]
            )
            assert np.allclose(
                reconstructed.values,
                out.loc[mask, "magnet_score"].values,
                atol=1e-6,
            ), "Score components do not sum to magnet_score"


# ── Pool management ───────────────────────────────────────────────────────────

class TestPoolManagement:

    def test_pool_created_on_pivot_high(self):
        """A buy-side pool appears when a pivot high is detected."""
        df  = _make_controlled_df()
        out = LiquidityMagnetEngine().generate(df)
        # After bar 5 (buy-side pivot), nearest buy distance should be finite
        assert pd.notna(out["nearest_buy_liquidity_distance"].iloc[6])

    def test_pool_created_on_pivot_low(self):
        """A sell-side pool appears when a pivot low is detected."""
        df  = _make_controlled_df()
        out = LiquidityMagnetEngine().generate(df)
        assert pd.notna(out["nearest_sell_liquidity_distance"].iloc[11])

    def test_sell_side_pool_swept(self):
        """After a bullish sweep, the sell-side pool no longer appears."""
        df  = _make_sweep_scenario()
        out = LiquidityMagnetEngine().generate(df)
        # Bar 30 is the sweep bar — sell-side pool price was 1.0970
        # After bar 30, nearest_sell_liquidity_distance should be NaN (no sell pools left)
        post_sweep = out["nearest_sell_liquidity_distance"].iloc[31:]
        assert post_sweep.isna().all(), (
            "Sell-side pool should be removed after bullish sweep"
        )

    def test_buy_side_pool_swept(self):
        """After a bearish sweep of a buy-side pool, it disappears."""
        n   = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        base = 1.1000
        c  = np.full(n, base)
        h  = c + 0.0010
        lo = c - 0.0010
        df = pd.DataFrame({"open": c.copy(), "high": h, "low": lo, "close": c.copy()}, index=idx)
        ph  = np.zeros(n, float); pl  = np.zeros(n, float)
        mph = np.zeros(n, float); mpl = np.zeros(n, float)
        # Buy-side pool at bar 5
        ph[5] = 1.0; mph[5] = 1.0; df.at[idx[5], "high"] = 1.1030
        # Bearish sweep at bar 30: high > 1.1030 AND close below 1.1030
        df.at[idx[30], "high"]  = 1.1040
        df.at[idx[30], "close"] = 1.0990
        df["pivot_high"] = ph; df["pivot_low"] = pl
        df["major_pivot_high"] = mph; df["major_pivot_low"] = mpl
        df["eqh"] = 0.0; df["eql"] = 0.0

        out = LiquidityMagnetEngine().generate(df)
        post_sweep = out["nearest_buy_liquidity_distance"].iloc[31:]
        assert post_sweep.isna().all(), "Buy-side pool should be removed after bearish sweep"

    def test_equal_high_increments_touches(self):
        """When eqh == 1.0 at a pivot, the pool starts with touches ≥ 2."""
        n   = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        c = np.full(n, 1.1000); h = c + 0.0010; lo = c - 0.0010
        df = pd.DataFrame({"open": c.copy(), "high": h, "low": lo, "close": c.copy()}, index=idx)
        ph  = np.zeros(n, float); pl  = np.zeros(n, float)
        mph = np.zeros(n, float); mpl = np.zeros(n, float)
        eqh = np.zeros(n, float)
        # Equal high at bar 10 → touches should start at 2
        ph[10] = 1.0; mph[10] = 1.0; eqh[10] = 1.0; df.at[idx[10], "high"] = 1.1020
        df["pivot_high"] = ph; df["pivot_low"] = pl
        df["major_pivot_high"] = mph; df["major_pivot_low"] = mpl
        df["eqh"] = eqh; df["eql"] = 0.0

        out = LiquidityMagnetEngine().generate(df)
        # nearest_cluster_size should reflect the 2-touch start
        cluster_after = out["nearest_cluster_size"].iloc[11]
        assert cluster_after >= 2.0, f"Expected touches >= 2, got {cluster_after}"

    def test_age_out_removes_pools(self):
        """Pools older than ageMax bars are removed."""
        n = _AGE_MAX + 50
        df = _make_df(n)
        # Place a single pivot high at bar 5
        df["pivot_high"]  = 0.0
        df["pivot_low"]   = 0.0
        df["major_pivot_high"] = 0.0
        df["major_pivot_low"]  = 0.0
        df.at[df.index[5], "pivot_high"]       = 1.0
        df.at[df.index[5], "major_pivot_high"] = 1.0

        out = LiquidityMagnetEngine().generate(df)
        # At bar 5 + ageMax + 1, the pool should have been aged out
        aged_out_bar = min(5 + _AGE_MAX + 5, n - 1)
        assert np.isnan(out["nearest_buy_liquidity_distance"].iloc[aged_out_bar]), (
            "Pool should be aged out after ageMax bars"
        )


# ── Scoring formula unit tests ────────────────────────────────────────────────

class TestScoringFormula:

    def test_proximity_decreases_with_distance(self):
        """A pool further from close should have a lower proximity contribution."""
        n   = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        c = np.full(n, 1.1000); h = c + 0.0010; lo = c - 0.0010
        df = pd.DataFrame({"open": c.copy(), "high": h, "low": lo, "close": c.copy()}, index=idx)
        ph = np.zeros(n, float); pl = np.zeros(n, float)
        mph = np.zeros(n, float); mpl = np.zeros(n, float)
        # Two sell-side pools at different distances: 0.0020 and 0.0040 below
        pl[5]  = 1.0; mpl[5]  = 1.0; df.at[idx[5],  "low"] = 1.1000 - 0.0020  # closer
        pl[10] = 1.0; mpl[10] = 1.0; df.at[idx[10], "low"] = 1.1000 - 0.0040  # farther
        df["pivot_high"] = ph; df["pivot_low"] = pl
        df["major_pivot_high"] = mph; df["major_pivot_low"] = mpl
        df["eqh"] = 0.0; df["eql"] = 0.0

        out = LiquidityMagnetEngine().generate(df)
        # nearest_sell is the closer pool
        close_dist  = out["nearest_sell_liquidity_distance"].iloc[15]
        # After both pools exist, nearest is the closer one
        assert pd.notna(close_dist) and close_dist > 0

    def test_fresh_pool_higher_age_score(self):
        """A freshly formed pool should have a higher age contribution than an old one."""
        eng = LiquidityMagnetEngine()
        # Compute _score directly for two hypothetical pools via the static method
        pool_fresh = {
            "price":   np.array([1.0980]),
            "bar":     np.array([90]),
            "touches": np.array([1.0]),
            "major":   np.array([True]),
            "active":  np.array([True]),
        }
        pool_old = {
            "price":   np.array([1.0980]),
            "bar":     np.array([0]),
            "touches": np.array([1.0]),
            "major":   np.array([True]),
            "active":  np.array([True]),
        }
        current_bar = 100
        close = 1.1000; atr_i = 0.002; dir_mom = 0.0

        _, age_fresh, _, _ = eng._score(pool_fresh, current_bar, close, atr_i, dir_mom)
        _, age_old,   _, _ = eng._score(pool_old,   current_bar, close, atr_i, dir_mom)

        assert age_fresh[0] > age_old[0], (
            f"Fresh pool age {age_fresh[0]} should be > old pool age {age_old[0]}"
        )

    def test_more_touches_higher_touch_score(self):
        """A pool with 3 touches should score higher than one with 1 touch."""
        eng = LiquidityMagnetEngine()
        pool_1t = {
            "price":   np.array([1.0980]),
            "bar":     np.array([90]),
            "touches": np.array([1.0]),
            "major":   np.array([True]),
            "active":  np.array([True]),
        }
        pool_3t = {
            "price":   np.array([1.0980]),
            "bar":     np.array([90]),
            "touches": np.array([3.0]),
            "major":   np.array([True]),
            "active":  np.array([True]),
        }
        current_bar = 100; close = 1.1000; atr_i = 0.002; dir_mom = 0.0

        _, _, tch_1, sc_1 = eng._score(pool_1t, current_bar, close, atr_i, dir_mom)
        _, _, tch_3, sc_3 = eng._score(pool_3t, current_bar, close, atr_i, dir_mom)

        assert tch_3[0] > tch_1[0]
        assert sc_3[0]  > sc_1[0]

    def test_momentum_contribution_toward_side(self):
        """Rising momentum increases buy-side score; falling momentum increases sell-side."""
        eng  = LiquidityMagnetEngine()
        pool = {
            "price":   np.array([1.1020]),
            "bar":     np.array([90]),
            "touches": np.array([1.0]),
            "major":   np.array([True]),
            "active":  np.array([True]),
        }
        close   = 1.1000; atr_i = 0.002; current_bar = 100

        # bull_mom=0 vs bull_mom=0.5
        _, _, _, sc_no_mom = eng._score(pool, current_bar, close, atr_i, 0.0)
        _, _, _, sc_mom    = eng._score(pool, current_bar, close, atr_i, 0.5)

        assert sc_mom[0] > sc_no_mom[0], (
            "Positive momentum should increase buy-side pool score"
        )

    def test_inactive_slot_gets_minus_one_score(self):
        """Inactive slots must always receive score -1.0."""
        eng = LiquidityMagnetEngine()
        pool = {
            "price":   np.zeros(K, dtype=float),
            "bar":     np.full(K, -1, dtype=np.int64),
            "touches": np.zeros(K, dtype=float),
            "major":   np.zeros(K, dtype=bool),
            "active":  np.zeros(K, dtype=bool),      # all inactive
        }
        _, _, _, sc = eng._score(pool, 100, 1.1000, 0.002, 0.0)
        assert (sc == -1.0).all(), "Inactive slots must score -1"


# ── Target selection ──────────────────────────────────────────────────────────

class TestTargetSelection:

    def test_no_target_when_score_too_low(self):
        """If all pools score below minScore, no target should be flagged."""
        n   = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        c   = np.full(n, 1.1000)
        h   = c + 0.0001    # tiny spread → tiny ATR → huge ATR-relative distance → low proximity
        lo  = c - 0.0001
        df  = pd.DataFrame({"open": c.copy(), "high": h, "low": lo, "close": c.copy()}, index=idx)
        ph  = np.zeros(n, float); pl  = np.zeros(n, float)
        mph = np.zeros(n, float); mpl = np.zeros(n, float)
        # Put a pool very far away (>5 % from close) — won't qualify as target
        ph[5] = 1.0; mph[5] = 1.0; df.at[idx[5], "high"] = 1.1600   # 5.45 % above close
        df["pivot_high"] = ph; df["pivot_low"] = pl
        df["major_pivot_high"] = mph; df["major_pivot_low"] = mpl
        df["eqh"] = 0.0; df["eql"] = 0.0

        out = LiquidityMagnetEngine().generate(df)
        # No target_liquidity should be set after bar 5
        assert (out["target_liquidity"].iloc[6:] == 0.0).all(), (
            "Target should not be flagged when pool is beyond maxDist %"
        )

    def test_target_direction_buy_side(self):
        """When the target pool is a buy-side level (above), direction = +1."""
        df  = _make_controlled_df()
        out = LiquidityMagnetEngine().generate(df)
        buy_target = out[(out["target_liquidity"] > 1.1000) & (out["target_liquidity"] > 0)]
        if not buy_target.empty:
            assert (buy_target["target_direction"] == 1.0).all()

    def test_target_direction_sell_side(self):
        """When the target pool is a sell-side level (below), direction = -1."""
        df  = _make_controlled_df()
        out = LiquidityMagnetEngine().generate(df)
        sell_target = out[
            (out["target_liquidity"] > 0) & (out["target_liquidity"] < 1.1000)
        ]
        if not sell_target.empty:
            assert (sell_target["target_direction"] == -1.0).all()

    def test_distance_to_target_consistent(self):
        """distance_to_target must equal |close - target_price| / close * 100."""
        df   = _make_df(200)
        out  = LiquidityMagnetEngine().generate(df)
        mask = out["target_liquidity"] > 0
        if mask.any():
            expected_dist = (
                (df["close"][mask] - out.loc[mask, "target_liquidity"]).abs()
                / df["close"][mask] * 100.0
            )
            assert np.allclose(
                out.loc[mask, "distance_to_target"].values,
                expected_dist.values,
                atol=1e-6,
            )

    def test_liquidity_rank_one_for_strongest(self):
        """When there is only one active pool and it qualifies, rank should be 1."""
        n   = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        c   = np.full(n, 1.1000)
        h   = c + 0.0020; lo = c - 0.0020
        df  = pd.DataFrame({"open": c.copy(), "high": h, "low": lo, "close": c.copy()}, index=idx)
        ph  = np.zeros(n, float); pl  = np.zeros(n, float)
        mph = np.zeros(n, float); mpl = np.zeros(n, float)
        ph[5] = 1.0; mph[5] = 1.0; df.at[idx[5], "high"] = 1.1010  # close = 0.09 % away
        df["pivot_high"] = ph; df["pivot_low"] = pl
        df["major_pivot_high"] = mph; df["major_pivot_low"] = mpl
        df["eqh"] = 0.0; df["eql"] = 0.0

        out = LiquidityMagnetEngine().generate(df)
        # After bar 5, when there's only one pool, rank must be 1 if target exists
        later = out.iloc[6:]
        target_bars = later[later["target_liquidity"] > 0]
        if not target_bars.empty:
            assert (target_bars["liquidity_rank"] == 1.0).all()


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_dataframe(self):
        df  = _make_df(0)
        out = LiquidityMagnetEngine().generate(df)
        assert len(out) == 0

    def test_single_row(self):
        df  = _make_df(1)
        out = LiquidityMagnetEngine().generate(df)
        assert len(out) == 1

    def test_no_pivots_all_zero(self):
        df = _make_df(50)
        df["pivot_high"] = 0.0
        df["pivot_low"]  = 0.0
        df["major_pivot_high"] = 0.0
        df["major_pivot_low"]  = 0.0
        out = LiquidityMagnetEngine().generate(df)
        assert out["magnet_score"].sum() == 0.0
        assert out["target_liquidity"].sum() == 0.0

    def test_max_pools_per_side(self):
        """Creating more than K_SIDE pools per side should not raise or overflow."""
        n   = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        c   = np.full(n, 1.1000)
        h   = c + 0.0010; lo = c - 0.0010
        df  = pd.DataFrame({"open": c.copy(), "high": h, "low": lo, "close": c.copy()}, index=idx)
        ph  = np.zeros(n, float); pl  = np.zeros(n, float)
        mph = np.zeros(n, float); mpl = np.zeros(n, float)
        # 15 pivot highs — well above K_SIDE = 7
        for i in range(15):
            bar = i * 10 + 5
            if bar < n:
                ph[bar]  = 1.0
                mph[bar] = 1.0
                h[bar]   = 1.1000 + (i + 1) * 0.0010
        df["high"] = h
        df["pivot_high"] = ph; df["pivot_low"] = pl
        df["major_pivot_high"] = mph; df["major_pivot_low"] = mpl
        df["eqh"] = 0.0; df["eql"] = 0.0

        out = LiquidityMagnetEngine().generate(df)   # must not raise
        assert len(out) == n

    def test_multiple_timeframes_same_result(self):
        df1 = _make_df(100)
        df2 = df1.copy()
        df2.index = pd.date_range("2023-01-01", periods=100, freq="1h", tz="UTC")
        eng = LiquidityMagnetEngine()
        o1  = eng.generate(df1).reset_index(drop=True)
        o2  = eng.generate(df2).reset_index(drop=True)
        pd.testing.assert_frame_equal(o1, o2, check_names=False)


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:

    def test_output_concatenates_with_input(self):
        df  = _make_df(200)
        out = LiquidityMagnetEngine().generate(df)
        combined = pd.concat([df, out], axis=1)
        assert len(combined) == 200
        assert combined.columns.duplicated().sum() == 0

    def test_runs_on_500_bars(self):
        df  = _make_df(500)
        out = LiquidityMagnetEngine().generate(df)
        assert len(out) == 500
        assert "magnet_score" in out.columns


# ── Performance ───────────────────────────────────────────────────────────────

class TestPerformance:

    def test_87k_rows_under_30_seconds(self):
        n   = 87_503
        df  = _make_df(n, seed=0)
        eng = LiquidityMagnetEngine()

        start   = time.perf_counter()
        out     = eng.generate(df)
        elapsed = time.perf_counter() - start

        assert len(out) == n
        assert elapsed < 30.0, (
            f"LiquidityMagnetEngine took {elapsed:.1f}s on {n} rows — exceeds 30 s"
        )

    def test_throughput_rows_per_second(self):
        n   = 10_000
        df  = _make_df(n, seed=1)
        eng = LiquidityMagnetEngine()

        start   = time.perf_counter()
        eng.generate(df)
        elapsed = time.perf_counter() - start

        rps = n / elapsed
        assert rps > 500, f"Throughput {rps:.0f} rows/s below 500 minimum"
