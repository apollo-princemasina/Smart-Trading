"""Tests for the Multi-Timeframe Feature Fusion Engine.

Coverage
--------
* TimeframeMapper            — normalise, prefix, timedelta, rank, hierarchy
* FeatureAligner             — column prefixing, no-look-ahead, edge cases
* FusionValidator            — timezone, monotonic, look-ahead, duplicates
* FeatureFusion              — full fusion, no-look-ahead, edge cases
* FusionEngine               — run, save/load, incremental, parallel, describe
* Performance                — 1-year M15 fusion in < 5 s
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.fusion.feature_alignment import (
    FeatureAligner,
    _BAR_OPEN_PREFIX,
)
from src.features.fusion.feature_fusion import FeatureFusion
from src.features.fusion.fusion_engine import FusionEngine
from src.features.fusion.fusion_validator import FusionValidator, ValidationResult
from src.features.fusion.timeframe_mapper import TimeframeMapper

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _m15(n: int = 96, start: str = "2024-01-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "open":       1.10 + rng.standard_normal(n) * 0.001,
            "close":      1.10 + rng.standard_normal(n) * 0.001,
            "volume":     rng.integers(1_000, 5_000, n).astype(float),
            "rsi":        rng.uniform(30, 70, n),
            "log_return": rng.normal(0, 0.001, n),
        },
        index=idx,
    )


def _h1(n: int = 24, start: str = "2024-01-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(43)
    return pd.DataFrame(
        {
            "close":  1.10 + rng.standard_normal(n) * 0.002,
            "volume": rng.integers(10_000, 50_000, n).astype(float),
            "rsi":    rng.uniform(30, 70, n),
            "trend":  rng.choice([-1.0, 0.0, 1.0], n),
        },
        index=idx,
    )


def _h4(n: int = 12, start: str = "2024-01-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(44)
    return pd.DataFrame(
        {
            "close":  1.10 + rng.standard_normal(n) * 0.003,
            "volume": rng.integers(50_000, 200_000, n).astype(float),
            "bias":   rng.choice([-1.0, 1.0], n).astype(float),
        },
        index=idx,
    )


def _daily(n: int = 5, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(45)
    return pd.DataFrame(
        {
            "close":  1.10 + rng.standard_normal(n) * 0.005,
            "volume": rng.integers(200_000, 1_000_000, n).astype(float),
            "range":  rng.uniform(0.005, 0.020, n),
        },
        index=idx,
    )


def _weekly(n: int = 3, start: str = "2023-12-25") -> pd.DataFrame:
    # Use 7-day intervals starting before M15 window for guaranteed coverage
    idx = pd.date_range(start, periods=n, freq="7D", tz="UTC")
    rng = np.random.default_rng(46)
    return pd.DataFrame(
        {
            "close": 1.10 + rng.standard_normal(n) * 0.010,
            "range": rng.uniform(0.020, 0.060, n),
        },
        index=idx,
    )


@pytest.fixture
def all_tfs() -> dict[str, pd.DataFrame]:
    return {
        "W":   _weekly(),
        "D":   _daily(),
        "H4":  _h4(),
        "H1":  _h1(),
        "M15": _m15(),
    }


@pytest.fixture
def engine(tmp_path: Path) -> FusionEngine:
    return FusionEngine(base_dir=tmp_path, cache=False)


@pytest.fixture
def engine_cached(tmp_path: Path) -> FusionEngine:
    return FusionEngine(base_dir=tmp_path, cache=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TimeframeMapper
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeframeMapperNormalise:
    def test_canonical_unchanged(self):
        for tf in ["W", "D", "H4", "H1", "M15"]:
            assert TimeframeMapper.normalise(tf) == tf

    def test_alias_4h(self):
        assert TimeframeMapper.normalise("4H") == "H4"

    def test_alias_1h(self):
        assert TimeframeMapper.normalise("1H") == "H1"

    def test_alias_1d(self):
        assert TimeframeMapper.normalise("1D") == "D"

    def test_alias_weekly(self):
        assert TimeframeMapper.normalise("weekly") == "W"

    def test_alias_15m(self):
        assert TimeframeMapper.normalise("15M") == "M15"

    def test_lowercase_canonical(self):
        assert TimeframeMapper.normalise("h1") == "H1"

    def test_unknown_returns_uppercase(self):
        assert TimeframeMapper.normalise("xyz") == "XYZ"


class TestTimeframeMapperMetadata:
    def test_prefixes(self):
        assert TimeframeMapper.prefix("W")   == "weekly"
        assert TimeframeMapper.prefix("D")   == "daily"
        assert TimeframeMapper.prefix("H4")  == "h4"
        assert TimeframeMapper.prefix("H1")  == "h1"
        assert TimeframeMapper.prefix("M15") == "m15"

    def test_timedelta_weekly(self):
        assert TimeframeMapper.timedelta("W") == pd.Timedelta(weeks=1)

    def test_timedelta_h1(self):
        assert TimeframeMapper.timedelta("H1") == pd.Timedelta(hours=1)

    def test_timedelta_m15(self):
        assert TimeframeMapper.timedelta("M15") == pd.Timedelta(minutes=15)

    def test_minutes(self):
        assert TimeframeMapper.minutes("H4")  == 240
        assert TimeframeMapper.minutes("M15") == 15

    def test_validate_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            TimeframeMapper.validate("X99")

    def test_is_valid_true(self):
        assert TimeframeMapper.is_valid("H4") is True

    def test_is_valid_false(self):
        assert TimeframeMapper.is_valid("X99") is False


class TestTimeframeMapperHierarchy:
    def test_rank_weekly_lowest_index(self):
        assert TimeframeMapper.rank("W") == 0

    def test_rank_m15_highest_index(self):
        assert TimeframeMapper.rank("M15") == 4

    def test_rank_ordering(self):
        ranks = [TimeframeMapper.rank(tf) for tf in ["W", "D", "H4", "H1", "M15"]]
        assert ranks == sorted(ranks)

    def test_is_higher_than(self):
        assert TimeframeMapper.is_higher_than("D", "H1") is True
        assert TimeframeMapper.is_higher_than("H1", "D") is False

    def test_higher_timeframes_from_h1(self):
        htf = TimeframeMapper.higher_timeframes("H1")
        assert htf == ["W", "D", "H4"]

    def test_higher_timeframes_from_m15(self):
        htf = TimeframeMapper.higher_timeframes("M15")
        assert htf == ["W", "D", "H4", "H1"]

    def test_lower_timeframes_from_w(self):
        ltf = TimeframeMapper.lower_timeframes("W")
        assert ltf == ["D", "H4", "H1", "M15"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. FeatureAligner — column prefixing
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureAlignerPrefixing:
    def test_h1_prefix_applied(self):
        aligner = FeatureAligner()
        m15_idx = pd.date_range("2024-01-01 01:00", periods=4, freq="15min", tz="UTC")
        h1_df   = _h1(n=2)
        result  = aligner.align(h1_df, "H1", m15_idx)
        assert all(c.startswith("h1_") or c.startswith(_BAR_OPEN_PREFIX)
                   for c in result.columns)

    def test_d_prefix_applied(self):
        aligner = FeatureAligner()
        m15_idx = pd.date_range("2024-01-02 00:00", periods=4, freq="15min", tz="UTC")
        d_df    = _daily(n=3)
        result  = aligner.align(d_df, "D", m15_idx)
        assert all(c.startswith("daily_") or c.startswith(_BAR_OPEN_PREFIX)
                   for c in result.columns)

    def test_m15_prefix_base(self):
        aligner = FeatureAligner()
        m15_df  = _m15(n=4)
        result  = aligner.prefix_base(m15_df)
        assert all(c.startswith("m15_") for c in result.columns)

    def test_index_matches_base_index(self):
        aligner = FeatureAligner()
        m15_idx = pd.date_range("2024-01-01 01:00", periods=8, freq="15min", tz="UTC")
        result  = aligner.align(_h1(n=4), "H1", m15_idx)
        assert (result.index == m15_idx).all()

    def test_shape_rows_match_base(self):
        aligner = FeatureAligner()
        m15_idx = pd.date_range("2024-01-01 01:00", periods=16, freq="15min", tz="UTC")
        result  = aligner.align(_h1(n=4), "H1", m15_idx)
        assert len(result) == 16

    def test_internal_bar_open_column_present(self):
        aligner = FeatureAligner()
        m15_idx = pd.date_range("2024-01-01 01:00", periods=4, freq="15min", tz="UTC")
        result  = aligner.align(_h1(n=2), "H1", m15_idx)
        assert f"{_BAR_OPEN_PREFIX}H1" in result.columns

    def test_drop_internal_removes_bar_open(self):
        aligner = FeatureAligner()
        m15_idx = pd.date_range("2024-01-01 01:00", periods=4, freq="15min", tz="UTC")
        result  = aligner.align(_h1(n=2), "H1", m15_idx)
        clean   = FeatureAligner.drop_internal_columns(result)
        assert not any(c.startswith(_BAR_OPEN_PREFIX) for c in clean.columns)

    def test_bar_open_columns_map(self):
        aligner = FeatureAligner()
        m15_idx = pd.date_range("2024-01-01 01:00", periods=4, freq="15min", tz="UTC")
        result  = aligner.align(_h1(n=2), "H1", m15_idx)
        col_map = FeatureAligner.bar_open_columns(result)
        assert "H1" in col_map


# ─────────────────────────────────────────────────────────────────────────────
# 3. FeatureAligner — no-look-ahead guarantee
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureAlignerNoLookAhead:
    """Verify the fundamental no-look-ahead guarantee with deterministic data."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        # H1 bars: 08:00 (val=100), 09:00 (val=200), 10:00 (val=300)
        h1_idx     = pd.date_range("2024-01-01 08:00", periods=3, freq="1h", tz="UTC")
        self.h1_df = pd.DataFrame({"feature": [100.0, 200.0, 300.0]}, index=h1_idx)
        # M15 bars: 09:00 → 10:45 (8 bars)
        self.m15_idx = pd.date_range("2024-01-01 09:00", periods=8, freq="15min", tz="UTC")
        self.aligner = FeatureAligner()

    def test_m15_09xx_uses_h1_08_not_09(self):
        """M15 09:00-09:45 must use H1 08:00 (val=100), not H1 09:00 (val=200)."""
        result = self.aligner.align(self.h1_df, "H1", self.m15_idx)
        for i in range(4):  # 09:00, 09:15, 09:30, 09:45
            assert result.iloc[i]["h1_feature"] == 100.0, (
                f"M15 {self.m15_idx[i]} should use H1 08:00 (val=100), "
                f"not H1 09:00 (val=200)"
            )

    def test_m15_10xx_uses_h1_09(self):
        """M15 10:00-10:45 must use H1 09:00 (val=200), not H1 10:00 (val=300)."""
        result = self.aligner.align(self.h1_df, "H1", self.m15_idx)
        for i in range(4, 8):  # 10:00, 10:15, 10:30, 10:45
            assert result.iloc[i]["h1_feature"] == 200.0

    def test_boundary_exactly_at_bar_close(self):
        """M15 bar exactly at H1 close time (10:00) uses that H1 bar (09:00)."""
        result = self.aligner.align(self.h1_df, "H1", self.m15_idx)
        assert result.loc[pd.Timestamp("2024-01-01 10:00", tz="UTC"), "h1_feature"] == 200.0

    def test_nan_before_first_htf_bar(self):
        """M15 bars before the first HTF available_at time produce NaN."""
        early_m15 = pd.date_range("2024-01-01 00:00", periods=8 * 8, freq="15min", tz="UTC")
        result    = self.aligner.align(self.h1_df, "H1", early_m15)
        # H1 bar at 08:00 available at 09:00; all M15 before 09:00 → NaN
        before_09 = result[result.index < pd.Timestamp("2024-01-01 09:00", tz="UTC")]
        assert before_09["h1_feature"].isna().all()

    def test_daily_alignment(self):
        """D1 bar at 2024-01-01 available for M15 bars on 2024-01-02+."""
        d_idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        d_df  = pd.DataFrame({"val": [10.0, 20.0, 30.0]}, index=d_idx)
        m15_idx = pd.date_range("2024-01-02 00:00", periods=4, freq="15min", tz="UTC")
        aligner = FeatureAligner()
        result  = aligner.align(d_df, "D", m15_idx)
        # D bar at 2024-01-01 → available_at = 2024-01-02 → val=10
        assert (result["daily_val"] == 10.0).all()

    def test_h4_alignment(self):
        """H4 bar at 08:00 available at 12:00; M15 bars 08:xx → 11:xx use prev H4."""
        h4_idx = pd.date_range("2024-01-01 00:00", periods=4, freq="4h", tz="UTC")
        h4_df  = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]}, index=h4_idx)
        m15_idx = pd.date_range("2024-01-01 08:00", periods=16, freq="15min", tz="UTC")
        aligner = FeatureAligner()
        result  = aligner.align(h4_df, "H4", m15_idx)
        # H4 bar 00:00 available at 04:00; H4 bar 04:00 available at 08:00
        # M15 at 08:00 → latest avail_at ≤ 08:00 → H4 04:00 (val=2)
        assert result.iloc[0]["h4_val"] == 2.0

    def test_multiple_m15_per_h1_same_value(self):
        """All 4 M15 bars within one H1 period share the same H1 feature."""
        h1_idx = pd.date_range("2024-01-01 00:00", periods=4, freq="1h", tz="UTC")
        h1_df  = pd.DataFrame({"v": [10.0, 20.0, 30.0, 40.0]}, index=h1_idx)
        m15_idx = pd.date_range("2024-01-01 01:00", periods=4, freq="15min", tz="UTC")
        aligner = FeatureAligner()
        result  = aligner.align(h1_df, "H1", m15_idx)
        # H1 at 00:00 available at 01:00; all 01:00-01:45 M15 → val=10
        assert result["h1_v"].nunique() == 1
        assert result.iloc[0]["h1_v"] == 10.0

    def test_weekly_alignment(self):
        """Weekly bar at 2024-01-01 available for M15 bars on 2024-01-08+."""
        w_idx  = pd.date_range("2024-01-01", periods=2, freq="7D", tz="UTC")
        w_df   = pd.DataFrame({"wval": [100.0, 200.0]}, index=w_idx)
        m15_idx = pd.date_range("2024-01-08 00:00", periods=4, freq="15min", tz="UTC")
        aligner = FeatureAligner()
        result  = aligner.align(w_df, "W", m15_idx)
        assert (result["weekly_wval"] == 100.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. FeatureAligner — edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureAlignerEdgeCases:
    def test_empty_htf_returns_nan_columns(self):
        aligner = FeatureAligner()
        empty   = pd.DataFrame(columns=["a", "b"])
        m15_idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
        result  = aligner.align(empty, "H1", m15_idx)
        # Empty HTF → prefixed NaN columns
        assert "h1_a" in result.columns
        assert result["h1_a"].isna().all()

    def test_single_m15_bar(self):
        aligner = FeatureAligner()
        h1_df   = _h1(n=3)
        m15_idx = pd.date_range("2024-01-01 03:00", periods=1, freq="15min", tz="UTC")
        result  = aligner.align(h1_df, "H1", m15_idx)
        assert len(result) == 1

    def test_duplicate_htf_timestamps_deduplicated(self):
        aligner = FeatureAligner()
        idx     = pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"],
            tz="UTC",
        )
        h1_df   = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
        m15_idx = pd.date_range("2024-01-01 01:00", periods=4, freq="15min", tz="UTC")
        result  = aligner.align(h1_df, "H1", m15_idx)
        assert len(result) == 4   # no crash, shape preserved

    def test_alias_normalised(self):
        aligner = FeatureAligner()
        m15_idx = pd.date_range("2024-01-01 04:00", periods=4, freq="15min", tz="UTC")
        result  = aligner.align(_h4(n=2), "4H", m15_idx)   # alias "4H" → "H4"
        assert any(c.startswith("h4_") for c in result.columns)

    def test_prefix_base_no_index_change(self):
        aligner = FeatureAligner()
        m15_df  = _m15(n=4)
        result  = aligner.prefix_base(m15_df)
        pd.testing.assert_index_equal(result.index, m15_df.index)

    def test_no_internal_col_for_m15(self):
        aligner = FeatureAligner()
        m15_df  = _m15(n=4)
        result  = aligner.prefix_base(m15_df)
        assert not any(c.startswith(_BAR_OPEN_PREFIX) for c in result.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FusionValidator — timezone
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionValidatorTimezone:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.v = FusionValidator()

    def test_utc_passes(self):
        df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
        r  = self.v.validate_timezone(df, "test")
        assert r.is_valid

    def test_naive_fails(self):
        df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3))
        r  = self.v.validate_timezone(df, "test")
        assert not r.is_valid
        assert any("timezone-naive" in e for e in r.errors)

    def test_non_datetime_index_fails(self):
        df = pd.DataFrame(index=range(3))
        r  = self.v.validate_timezone(df, "test")
        assert not r.is_valid

    def test_non_utc_warns(self):
        df = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=3, tz="Europe/London")
        )
        r  = self.v.validate_timezone(df, "test")
        assert r.is_valid          # warns, not errors
        assert len(r.warnings) > 0

    def test_validate_all_propagates_tz_errors(self):
        naive_df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3))
        utc_df   = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
        r = self.v.validate_all({"M15": utc_df, "H1": naive_df})
        assert not r.is_valid


# ─────────────────────────────────────────────────────────────────────────────
# 6. FusionValidator — monotonic
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionValidatorMonotonic:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.v = FusionValidator()

    def test_monotonic_passes(self):
        df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=5, tz="UTC"))
        r  = self.v.validate_monotonic(df, "test")
        assert r.is_valid

    def test_non_monotonic_fails(self):
        idx = pd.DatetimeIndex(
            ["2024-01-02", "2024-01-01", "2024-01-03"], tz="UTC"
        )
        df  = pd.DataFrame(index=idx)
        r   = self.v.validate_monotonic(df, "test")
        assert not r.is_valid

    def test_duplicate_timestamps_fail(self):
        idx = pd.DatetimeIndex(
            ["2024-01-01", "2024-01-01", "2024-01-02"], tz="UTC"
        )
        df  = pd.DataFrame(index=idx)
        r   = self.v.validate_monotonic(df, "test")
        assert not r.is_valid
        assert any("duplicate" in e.lower() for e in r.errors)

    def test_single_row_passes(self):
        df = pd.DataFrame(index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"))
        r  = self.v.validate_monotonic(df, "test")
        assert r.is_valid


# ─────────────────────────────────────────────────────────────────────────────
# 7. FusionValidator — look-ahead detection
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionValidatorLookAhead:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.v = FusionValidator()

    def test_correct_alignment_passes(self):
        """H1 bar at 08:00 (avail=09:00) used for M15 bars at 09:00+ → no violation."""
        m15_idx = pd.date_range("2024-01-01 09:00", periods=4, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "h1_feat": [100.0] * 4,
                f"{_BAR_OPEN_PREFIX}H1": pd.DatetimeIndex(
                    ["2024-01-01 08:00"] * 4, tz="UTC"
                ),
            },
            index=m15_idx,
        )
        r = self.v.validate_no_lookahead(df)
        assert r.is_valid
        assert r.stats.get("H1_lookahead_violations") == 0

    def test_future_htf_bar_detected(self):
        """H1 bar at 09:00 (avail=10:00) used for M15 bars at 09:xx → violation."""
        m15_idx = pd.date_range("2024-01-01 09:00", periods=4, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "h1_feat": [200.0] * 4,
                f"{_BAR_OPEN_PREFIX}H1": pd.DatetimeIndex(
                    ["2024-01-01 09:00"] * 4, tz="UTC"  # closes at 10:00 → look-ahead!
                ),
            },
            index=m15_idx,
        )
        r = self.v.validate_no_lookahead(df)
        assert not r.is_valid
        assert any("look-ahead" in e for e in r.errors)

    def test_nat_rows_skipped(self):
        """NaT rows in bar-open column should not cause false look-ahead errors."""
        m15_idx = pd.date_range("2024-01-01 00:00", periods=4, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "h1_feat": [np.nan] * 4,
                f"{_BAR_OPEN_PREFIX}H1": [pd.NaT] * 4,
            },
            index=m15_idx,
        )
        r = self.v.validate_no_lookahead(df)
        assert r.is_valid   # all NaT → no violations detectable

    def test_mixed_valid_and_violation(self):
        """Only the violating rows should be counted."""
        m15_idx = pd.date_range("2024-01-01 09:00", periods=4, freq="15min", tz="UTC")
        bar_opens = pd.array(
            [
                pd.Timestamp("2024-01-01 08:00", tz="UTC"),  # avail=09:00 ≤ 09:00 ✓
                pd.Timestamp("2024-01-01 08:00", tz="UTC"),  # ✓
                pd.Timestamp("2024-01-01 09:00", tz="UTC"),  # avail=10:00 > 09:30 ✗
                pd.Timestamp("2024-01-01 09:00", tz="UTC"),  # ✗
            ]
        )
        df = pd.DataFrame(
            {
                "h1_feat": [100.0, 100.0, 200.0, 200.0],
                f"{_BAR_OPEN_PREFIX}H1": bar_opens,
            },
            index=m15_idx,
        )
        r = self.v.validate_no_lookahead(df)
        assert not r.is_valid
        assert any("2 M15 bar" in e for e in r.errors)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FusionValidator — completeness and duplicates
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionValidatorCompleteness:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.v = FusionValidator()

    def test_zero_nan_returns_zero(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]},
                          index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        pct = self.v.validate_completeness(df)
        assert pct["a"] == pytest.approx(0.0)

    def test_all_nan_returns_100(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]},
                          index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        pct = self.v.validate_completeness(df)
        assert pct["a"] == pytest.approx(100.0)

    def test_partial_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]},
                          index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"))
        pct = self.v.validate_completeness(df)
        assert pct["a"] == pytest.approx(50.0)

    def test_empty_df_returns_empty(self):
        pct = self.v.validate_completeness(pd.DataFrame())
        assert pct == {}

    def test_duplicate_columns_detected(self):
        groups = {
            "M15": ["m15_close", "m15_rsi"],
            "H1":  ["h1_close",  "m15_close"],   # duplicate m15_close
        }
        r = self.v.validate_no_duplicates(groups)
        assert not r.is_valid
        assert any("m15_close" in e for e in r.errors)

    def test_no_duplicates_passes(self):
        groups = {
            "M15": ["m15_close"],
            "H1":  ["h1_close"],
        }
        r = self.v.validate_no_duplicates(groups)
        assert r.is_valid


# ─────────────────────────────────────────────────────────────────────────────
# 9. ValidationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationResult:
    def test_default_valid(self):
        r = ValidationResult()
        assert r.is_valid is True
        assert r.errors == []

    def test_add_error_invalidates(self):
        r = ValidationResult()
        r.add_error("oops")
        assert not r.is_valid
        assert "oops" in r.errors

    def test_merge_propagates_invalid(self):
        a = ValidationResult()
        b = ValidationResult()
        b.add_error("fail")
        a.merge(b)
        assert not a.is_valid

    def test_raise_if_invalid(self):
        r = ValidationResult()
        r.add_error("bad")
        with pytest.raises(ValueError, match="bad"):
            r.raise_if_invalid()

    def test_raise_if_valid_does_not_raise(self):
        r = ValidationResult()
        r.raise_if_invalid()   # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# 10. FeatureFusion — full fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureFusionFull:
    @pytest.fixture(autouse=True)
    def _setup(self, all_tfs):
        self.tfs    = all_tfs
        self.fusion = FeatureFusion(validate=True)

    def test_returns_dataframe(self):
        fused, _ = self.fusion.fuse(self.tfs)
        assert isinstance(fused, pd.DataFrame)

    def test_index_matches_m15(self):
        fused, _ = self.fusion.fuse(self.tfs)
        pd.testing.assert_index_equal(fused.index, self.tfs["M15"].index)

    def test_all_prefixes_present(self):
        fused, _ = self.fusion.fuse(self.tfs)
        prefixes = {col.split("_")[0] for col in fused.columns if "_" in col}
        for expected in ("m15", "h1", "h4", "daily", "weekly"):
            assert expected in prefixes, f"Prefix '{expected}' missing"

    def test_m15_prefix_columns_exist(self):
        fused, _ = self.fusion.fuse(self.tfs)
        m15_cols  = [c for c in fused.columns if c.startswith("m15_")]
        assert len(m15_cols) == len(self.tfs["M15"].columns)

    def test_validation_result_returned(self):
        _, val = self.fusion.fuse(self.tfs)
        assert isinstance(val, ValidationResult)

    def test_no_internal_columns_by_default(self):
        fused, _ = self.fusion.fuse(self.tfs, drop_internal=True)
        assert not any(c.startswith(_BAR_OPEN_PREFIX) for c in fused.columns)

    def test_internal_columns_kept_when_requested(self):
        fused, _ = self.fusion.fuse(self.tfs, drop_internal=False)
        internal = [c for c in fused.columns if c.startswith(_BAR_OPEN_PREFIX)]
        assert len(internal) > 0

    def test_column_count_reasonable(self):
        fused, _ = self.fusion.fuse(self.tfs)
        # m15=5, h1=4, h4=3, daily=3, weekly=2 → at least 17
        assert fused.shape[1] >= 17

    def test_m15_required_raises(self):
        fusion = FeatureFusion()
        with pytest.raises(KeyError, match="M15"):
            fusion.fuse({"H1": _h1()})

    def test_describe_output_structure(self):
        fused, _ = self.fusion.fuse(self.tfs)
        desc     = self.fusion.describe(fused)
        assert desc["n_rows"] == len(self.tfs["M15"])
        assert "cols_by_prefix" in desc
        assert "nan_pct" in desc


# ─────────────────────────────────────────────────────────────────────────────
# 11. FeatureFusion — no-look-ahead (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureFusionNoLookAhead:
    """End-to-end verification that the fused output contains no look-ahead."""

    def test_h1_values_correct(self):
        """
        H1 bars 00:00-23:00 with vals 100-123.
        M15 bar at 01:00 must receive val=100 (H1 00:00 available at 01:00).
        M15 bar at 02:00 must receive val=101 (H1 01:00 available at 02:00).
        M15 bar at 00:00 must be NaN (no completed H1 bar yet).
        """
        h1_idx = pd.date_range("2024-01-01 00:00", periods=24, freq="1h", tz="UTC")
        h1_df  = pd.DataFrame({"v": list(range(100, 124))}, index=h1_idx)
        m15_idx = pd.date_range("2024-01-01 00:00", periods=96, freq="15min", tz="UTC")
        m15_df  = pd.DataFrame({"close": 1.1}, index=m15_idx)

        fusion = FeatureFusion(validate=True)
        fused, val = fusion.fuse({"M15": m15_df, "H1": h1_df})

        # NaN before first H1 bar closes (00:00-00:45)
        assert pd.isna(fused.loc[m15_idx[0], "h1_v"])
        assert pd.isna(fused.loc[m15_idx[3], "h1_v"])

        # val=100 from 01:00 to 01:45
        assert fused.loc[m15_idx[4], "h1_v"] == 100.0
        assert fused.loc[m15_idx[7], "h1_v"] == 100.0

        # val=101 from 02:00 onward
        assert fused.loc[m15_idx[8], "h1_v"] == 101.0

    def test_validator_reports_no_lookahead(self):
        fusion = FeatureFusion(validate=True)
        fused_with_internal, _ = fusion.fuse(
            {"M15": _m15(), "H1": _h1()},
            drop_internal=False,
        )
        v = FusionValidator()
        r = v.validate_no_lookahead(fused_with_internal)
        assert r.is_valid, f"Look-ahead violations: {r.errors}"

    def test_partial_timeframes_no_lookahead(self):
        """M15-only fusion produces no look-ahead (trivially)."""
        fusion = FeatureFusion(validate=True)
        fused, val = fusion.fuse({"M15": _m15()})
        assert val.is_valid

    def test_h4_no_lookahead(self):
        """Verify H4 context never uses a future H4 bar."""
        fusion = FeatureFusion(validate=True)
        _, val = fusion.fuse(
            {"M15": _m15(), "H4": _h4()},
            drop_internal=False,
        )
        assert not any("look-ahead" in e for e in val.errors), val.errors

    def test_daily_no_lookahead(self):
        fusion = FeatureFusion(validate=True)
        _, val = fusion.fuse(
            {"M15": _m15(), "D": _daily()},
            drop_internal=False,
        )
        assert not any("look-ahead" in e for e in val.errors), val.errors


# ─────────────────────────────────────────────────────────────────────────────
# 12. FeatureFusion — edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureFusionEdgeCases:
    def test_m15_only(self):
        fusion = FeatureFusion()
        fused, _ = fusion.fuse({"M15": _m15(n=8)})
        m15_cols = [c for c in fused.columns if c.startswith("m15_")]
        assert len(m15_cols) > 0
        assert not any(c.startswith("h1_") for c in fused.columns)

    def test_missing_htf_silently_skipped(self):
        fusion = FeatureFusion()
        # No W or D provided; should still produce h1_ and m15_ columns
        fused, _ = fusion.fuse({"M15": _m15(), "H1": _h1()})
        assert not any(c.startswith("weekly_") for c in fused.columns)
        assert not any(c.startswith("daily_")  for c in fused.columns)
        assert any(c.startswith("h1_") for c in fused.columns)

    def test_alias_keys_accepted(self):
        fusion = FeatureFusion()
        fused, _ = fusion.fuse({"M15": _m15(), "1H": _h1(), "4H": _h4()})
        assert any(c.startswith("h1_") for c in fused.columns)
        assert any(c.startswith("h4_") for c in fused.columns)

    def test_empty_htf_produces_nan_cols(self):
        fusion = FeatureFusion(validate=False)
        empty_h1 = pd.DataFrame(columns=["close", "rsi"])
        fused, _ = fusion.fuse({"M15": _m15(n=4), "H1": empty_h1})
        assert any(c.startswith("h1_") for c in fused.columns)
        h1_cols = [c for c in fused.columns if c.startswith("h1_")]
        assert fused[h1_cols].isna().all().all()

    def test_strict_mode_raises_on_invalid(self):
        fusion     = FeatureFusion(validate=True, strict=True)
        naive_m15  = pd.DataFrame(
            {"close": [1.1]},
            index=pd.DatetimeIndex(["2024-01-01"]),  # tz-naive
        )
        with pytest.raises(ValueError):
            fusion.fuse({"M15": naive_m15})


# ─────────────────────────────────────────────────────────────────────────────
# 13. FusionEngine — run
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionEngineRun:
    def test_run_returns_dataframe(self, engine, all_tfs):
        fused, _ = engine.run("EURUSD", all_tfs, save=False)
        assert isinstance(fused, pd.DataFrame)
        assert len(fused) == len(all_tfs["M15"])

    def test_run_saves_parquet(self, engine, tmp_path, all_tfs):
        engine.run("EURUSD", all_tfs, save=True)
        expected = tmp_path / "EURUSD" / "feature_dataset_fused.parquet"
        assert expected.exists()

    def test_saved_parquet_readable(self, engine, tmp_path, all_tfs):
        fused, _ = engine.run("EURUSD", all_tfs, save=True)
        loaded   = pd.read_parquet(tmp_path / "EURUSD" / "feature_dataset_fused.parquet")
        assert loaded.shape == fused.shape

    def test_run_without_save(self, engine, tmp_path, all_tfs):
        engine.run("EURUSD", all_tfs, save=False)
        assert not (tmp_path / "EURUSD" / "feature_dataset_fused.parquet").exists()

    def test_on_complete_callback(self, engine, all_tfs):
        called_with = []
        engine.run("EURUSD", all_tfs, save=False, on_complete=called_with.append)
        assert len(called_with) == 1
        assert isinstance(called_with[0], pd.DataFrame)

    def test_load_nonexistent_returns_none(self, engine):
        assert engine.load("NONEXISTENT") is None

    def test_load_existing_returns_df(self, engine, all_tfs):
        fused, _ = engine.run("EURUSD", all_tfs, save=True)
        loaded   = engine.load("EURUSD")
        assert loaded is not None
        assert len(loaded) == len(fused)

    def test_cache_hit_skips_recomputation(self, engine_cached, all_tfs):
        fused1, _ = engine_cached.run("EURUSD", all_tfs, save=True)
        fused2, _ = engine_cached.run("EURUSD", all_tfs, save=False)
        assert fused1.shape == fused2.shape


# ─────────────────────────────────────────────────────────────────────────────
# 14. FusionEngine — incremental update
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionEngineIncremental:
    @pytest.fixture(autouse=True)
    def _base_dataset(self, engine, all_tfs):
        """Run full fusion first to create the initial cached file."""
        self.engine  = engine
        self.all_tfs = all_tfs
        self.fused0, _ = engine.run("EURUSD", all_tfs, save=True)

    def test_incremental_appends_new_bars(self):
        # Create 4 new M15 bars following the original dataset
        last_ts = self.all_tfs["M15"].index[-1]
        new_m15 = pd.DataFrame(
            {"open": 1.1, "close": 1.1, "volume": 1000.0,
             "rsi": 50.0, "log_return": 0.0},
            index=pd.date_range(
                last_ts + pd.Timedelta(minutes=15), periods=4, freq="15min", tz="UTC"
            ),
        )
        new_tfs = dict(self.all_tfs)
        new_tfs["M15"] = pd.concat([self.all_tfs["M15"], new_m15])
        combined, _ = self.engine.update_incremental("EURUSD", new_tfs, save=True)
        assert len(combined) == len(self.fused0) + 4

    def test_incremental_no_duplicates(self):
        new_tfs = dict(self.all_tfs)
        combined, _ = self.engine.update_incremental("EURUSD", new_tfs, save=False)
        assert not combined.index.duplicated().any()

    def test_incremental_no_new_bars_returns_existing(self):
        # Same M15 data → no new bars
        combined, _ = self.engine.update_incremental(
            "EURUSD", self.all_tfs, save=False
        )
        assert len(combined) == len(self.fused0)

    def test_incremental_returns_sorted_index(self):
        last_ts = self.all_tfs["M15"].index[-1]
        new_m15 = pd.DataFrame(
            {"open": 1.1, "close": 1.1, "volume": 1000.0,
             "rsi": 50.0, "log_return": 0.0},
            index=pd.date_range(
                last_ts + pd.Timedelta(minutes=15), periods=2, freq="15min", tz="UTC"
            ),
        )
        new_tfs = dict(self.all_tfs)
        new_tfs["M15"] = pd.concat([self.all_tfs["M15"], new_m15])
        combined, _ = self.engine.update_incremental("EURUSD", new_tfs, save=False)
        assert combined.index.is_monotonic_increasing

    def test_incremental_full_fusion_on_first_call(self, tmp_path):
        fresh = FusionEngine(base_dir=tmp_path / "fresh", cache=False)
        combined, _ = fresh.update_incremental("EURUSD", self.all_tfs, save=True)
        assert len(combined) == len(self.fused0)


# ─────────────────────────────────────────────────────────────────────────────
# 15. FusionEngine — parallel multi-symbol
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionEngineParallel:
    @pytest.fixture(autouse=True)
    def _setup(self, engine):
        self.engine = engine

    def test_run_many_sequential(self, all_tfs):
        results = self.engine.run_many(
            {"EURUSD": all_tfs, "GBPUSD": all_tfs},
            save=False,
            parallel=False,
        )
        assert set(results) == {"EURUSD", "GBPUSD"}
        for sym, (df, _) in results.items():
            assert len(df) == len(all_tfs["M15"])

    def test_run_many_parallel(self, all_tfs):
        results = self.engine.run_many(
            {"EURUSD": all_tfs, "GBPUSD": all_tfs, "USDJPY": all_tfs},
            save=False,
            parallel=True,
        )
        assert len(results) == 3
        for _, (df, _) in results.items():
            assert isinstance(df, pd.DataFrame)

    def test_parallel_results_match_sequential(self, all_tfs):
        par = self.engine.run_many({"EURUSD": all_tfs}, save=False, parallel=True)
        seq = self.engine.run_many({"EURUSD": all_tfs}, save=False, parallel=False)
        pd.testing.assert_frame_equal(par["EURUSD"][0], seq["EURUSD"][0])

    def test_run_many_saves_all(self, tmp_path, all_tfs):
        eng = FusionEngine(base_dir=tmp_path, cache=False)
        eng.run_many({"EURUSD": all_tfs, "GBPUSD": all_tfs}, save=True)
        assert (tmp_path / "EURUSD" / "feature_dataset_fused.parquet").exists()
        assert (tmp_path / "GBPUSD" / "feature_dataset_fused.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 16. FusionEngine — describe
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionEngineDescribe:
    def test_describe_not_found(self, engine):
        info = engine.describe("NONEXISTENT")
        assert info["status"] == "not_found"

    def test_describe_existing(self, engine, all_tfs):
        engine.run("EURUSD", all_tfs, save=True)
        info = engine.describe("EURUSD")
        assert info["n_rows"] == len(all_tfs["M15"])
        assert "n_cols" in info
        assert "start" in info
        assert "end" in info
        assert "path" in info

    def test_describe_columns_listed(self, engine, all_tfs):
        engine.run("EURUSD", all_tfs, save=True)
        info = engine.describe("EURUSD")
        assert isinstance(info["columns"], list)
        assert any("m15_" in c for c in info["columns"])


# ─────────────────────────────────────────────────────────────────────────────
# 17. Performance
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:
    @pytest.mark.slow
    def test_fusion_1yr_m15_under_5s(self):
        """Full 5-TF fusion of 1 year of M15 data should complete in < 5 s."""
        n_m15   = 365 * 24 * 4   # ~35 040 bars
        n_h1    = 365 * 24
        n_h4    = 365 * 6
        n_daily = 366
        n_weekly = 54

        m15_df = _m15(n_m15)
        h1_df  = _h1(n_h1)
        h4_df  = _h4(n_h4)
        d_df   = _daily(n_daily)
        w_df   = _weekly(n_weekly, start="2023-01-01")

        fusion = FeatureFusion(validate=False)   # skip validation for pure speed

        t0 = time.perf_counter()
        fused, _ = fusion.fuse(
            {"M15": m15_df, "H1": h1_df, "H4": h4_df, "D": d_df, "W": w_df}
        )
        elapsed = time.perf_counter() - t0

        assert len(fused) == n_m15, "Row count mismatch"
        assert elapsed < 5.0, f"Fusion took {elapsed:.2f}s (limit 5s)"

    def test_fusion_shape_5tf(self):
        """Verify that 5-TF fusion produces the expected number of columns."""
        fusion = FeatureFusion(validate=False)
        tfs = {
            "W":   _weekly(),
            "D":   _daily(),
            "H4":  _h4(),
            "H1":  _h1(),
            "M15": _m15(),
        }
        fused, _ = fusion.fuse(tfs)
        # m15=5, h1=4, h4=3, daily=3, weekly=2 = 17 minimum
        assert fused.shape[1] >= 17
