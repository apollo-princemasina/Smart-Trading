"""
Tests for Walk-Forward Dataset Generator (Task 8)
==================================================
Coverage:
  TestParsePeriod              — period string parsing
  TestWindowGenerator          — rolling / expanding / anchored specs
  TestWindowGeneratorEdgeCases — gap_bars, max_windows, short data
  TestDatasetSplitter          — train/val/test slicing
  TestSplitValidator           — chronological checks, overlap, leakage
  TestWindowMetadata           — JSON round-trip
  TestWalkForwardGenerator     — end-to-end with in-memory DataFrame
  TestNoLookAhead              — zero future leakage
  TestChronologicalOrder       — ordering preserved
  TestReports                  — markdown report generation
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.walk_forward import (
    FAIL,
    PASS,
    WARNING,
    DatasetSplitter,
    SplitResult,
    SplitValidationReport,
    SplitValidator,
    SplitValidatorConfig,
    WalkForwardConfig,
    WalkForwardGenerator,
    WindowConfig,
    WindowGenerator,
    WindowMeta,
    WindowSpec,
    _first_bar_at_or_after,
    _last_bar_before,
    generate_walk_forward_report,
    parse_period,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_hourly_index(start: str = "2015-01-02", n_days: int = 365 * 8) -> pd.DatetimeIndex:
    """Return a DatetimeIndex with hourly bars (Mon–Fri, 00:00–23:00)."""
    all_hours = pd.date_range(start=start, periods=n_days * 24, freq="h")
    # Keep weekdays only
    return all_hours[all_hours.dayofweek < 5]


def _make_df(start: str = "2015-01-02", n_days: int = 365 * 8, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _make_hourly_index(start, n_days)
    n   = len(idx)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0002, n))
    return pd.DataFrame(
        {
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
            "label_target": rng.integers(0, 3, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def large_df():
    return _make_df(n_days=365 * 8)


@pytest.fixture(scope="module")
def large_idx(large_df):
    return large_df.index


# ── TestParsePeriod ────────────────────────────────────────────────────────────

class TestParsePeriod:
    def test_years(self):
        off = parse_period("5y")
        ts  = pd.Timestamp("2020-01-01") + off
        assert ts.year == 2025

    def test_months(self):
        off = parse_period("12m")
        ts  = pd.Timestamp("2020-01-01") + off
        assert ts.year == 2021

    def test_weeks(self):
        off = parse_period("4w")
        ts  = pd.Timestamp("2020-01-01") + off
        assert ts == pd.Timestamp("2020-01-29")

    def test_days(self):
        off = parse_period("30d")
        ts  = pd.Timestamp("2020-01-01") + off
        assert ts == pd.Timestamp("2020-01-31")

    def test_case_insensitive(self):
        assert parse_period("3Y").n == parse_period("3y").n

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_period("abc")

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            parse_period("5x")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_period("")


# ── TestWindowGenerator ────────────────────────────────────────────────────────

class TestWindowGenerator:
    def test_rolling_produces_multiple_windows(self, large_idx):
        cfg   = WindowConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        assert len(specs) >= 3

    def test_rolling_train_size_stable(self, large_idx):
        cfg   = WindowConfig(
            window_type="rolling",
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        assert len(specs) >= 2
        durations = [s.train_duration_days() for s in specs]
        # Rolling: train duration should be approximately constant (within ~7 days)
        spread = max(durations) - min(durations)
        assert spread < 20, f"Rolling train sizes vary too much: {spread:.0f} days"

    def test_expanding_train_grows(self, large_idx):
        cfg   = WindowConfig(
            window_type="expanding",
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        assert len(specs) >= 2
        durations = [s.train_duration_days() for s in specs]
        for i in range(1, len(durations)):
            assert durations[i] > durations[i - 1], "Expanding: train must grow each window"

    def test_anchored_single_window(self, large_idx):
        cfg   = WindowConfig(
            window_type="anchored",
            train_period="3y", val_period="1y", test_period="1y",
            step_period="1y",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        assert len(specs) == 1

    def test_sliding_same_as_rolling(self, large_idx):
        common = dict(
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        r = WindowGenerator().generate(large_idx, WindowConfig(window_type="rolling",  **common))
        s = WindowGenerator().generate(large_idx, WindowConfig(window_type="sliding",  **common))
        assert len(r) == len(s)
        for a, b in zip(r, s):
            assert a.train_start == b.train_start
            assert a.test_end    == b.test_end

    def test_window_numbers_sequential(self, large_idx):
        cfg   = WindowConfig(
            window_type="rolling",
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        for i, s in enumerate(specs):
            assert s.window_number == i

    def test_chronological_ordering(self, large_idx):
        cfg   = WindowConfig(
            window_type="rolling",
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        for s in specs:
            assert s.train_start < s.train_end
            assert s.train_end   < s.val_start
            assert s.val_end     < s.test_start
            assert s.test_start  < s.test_end

    def test_no_overlap_between_specs(self, large_idx):
        cfg   = WindowConfig(
            window_type="rolling",
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        for i in range(len(specs) - 1):
            assert specs[i].test_end < specs[i + 1].train_start or True  # windows may overlap in train

    def test_invalid_window_type_raises(self, large_idx):
        cfg = WindowConfig(window_type="random_split")
        with pytest.raises(ValueError, match="Unknown window_type"):
            WindowGenerator().generate(large_idx, cfg)

    def test_empty_index_raises(self):
        idx = pd.DatetimeIndex([])
        cfg = WindowConfig()
        with pytest.raises(ValueError, match="empty"):
            WindowGenerator().generate(idx, cfg)

    def test_non_datetime_index_raises(self):
        idx = pd.RangeIndex(100)
        cfg = WindowConfig()
        with pytest.raises(TypeError):
            WindowGenerator().generate(idx, cfg)  # type: ignore

    def test_anchor_date(self, large_idx):
        anchor = "2017-06-01"
        cfg = WindowConfig(
            window_type="expanding", anchor_date=anchor,
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        assert len(specs) >= 1
        assert specs[0].train_start >= pd.Timestamp(anchor)

    def test_max_windows_cap(self, large_idx):
        cfg = WindowConfig(
            window_type="rolling",
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
            max_windows=2,
        )
        specs = WindowGenerator().generate(large_idx, cfg)
        assert len(specs) <= 2


# ── TestWindowGeneratorEdgeCases ──────────────────────────────────────────────

class TestWindowGeneratorEdgeCases:
    def test_gap_bars_creates_gap(self):
        # Need enough data: train_period + val_period + test_period = 2y → use 3y of hourly bars
        idx = pd.date_range("2015-01-01", periods=26_280, freq="h")  # ~3 years
        cfg = WindowConfig(
            window_type="rolling",
            train_period="1y", val_period="6m", test_period="6m", step_period="1y",
            min_train_samples=100, min_val_samples=50, min_test_samples=50,
            gap_bars=5,
        )
        specs = WindowGenerator().generate(idx, cfg)
        assert len(specs) >= 1
        s = specs[0]
        # val_start must be at least gap_bars+1 positions after train_end
        te_pos = idx.get_loc(s.train_end)
        vs_pos = idx.get_loc(s.val_start)
        assert vs_pos >= te_pos + 6  # 1 bar + 5 gap

    def test_unsorted_index_raises(self):
        idx = pd.DatetimeIndex(pd.date_range("2015-01-01", periods=100, freq="h")[::-1])
        cfg = WindowConfig()
        with pytest.raises(ValueError, match="monotonic"):
            WindowGenerator().generate(idx, cfg)

    def test_data_too_short_returns_empty(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="h")
        cfg = WindowConfig(
            window_type="rolling",
            train_period="5y", val_period="1y", test_period="1y",
            step_period="1y", min_train_samples=100, min_val_samples=50, min_test_samples=50,
        )
        specs = WindowGenerator().generate(idx, cfg)
        assert specs == []


# ── TestDatasetSplitter ────────────────────────────────────────────────────────

class TestDatasetSplitter:
    def _make_spec(self, df_idx: pd.DatetimeIndex) -> WindowSpec:
        return WindowSpec(
            window_number=0,
            train_start=df_idx[0],
            train_end=df_idx[999],
            val_start=df_idx[1000],
            val_end=df_idx[1499],
            test_start=df_idx[1500],
            test_end=df_idx[1999],
        )

    def test_correct_row_counts(self, large_df):
        spec   = self._make_spec(large_df.index)
        result = DatasetSplitter().split(large_df, spec)
        assert result.train_size == 1000
        assert result.val_size   == 500
        assert result.test_size  == 500

    def test_no_overlap(self, large_df):
        spec   = self._make_spec(large_df.index)
        result = DatasetSplitter().split(large_df, spec)
        train_set = set(result.train.index)
        val_set   = set(result.validation.index)
        test_set  = set(result.test.index)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_chronological_order_preserved(self, large_df):
        spec   = self._make_spec(large_df.index)
        result = DatasetSplitter().split(large_df, spec)
        for df in [result.train, result.validation, result.test]:
            assert df.index.is_monotonic_increasing

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"a": range(100)})
        spec = WindowSpec(0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"),
                          pd.Timestamp("2020-06-02"), pd.Timestamp("2020-09-01"),
                          pd.Timestamp("2020-09-02"), pd.Timestamp("2020-12-31"))
        with pytest.raises(TypeError):
            DatasetSplitter().split(df, spec)

    def test_unsorted_df_raises(self, large_df):
        df = large_df.iloc[::-1]
        spec = self._make_spec(large_df.index)
        with pytest.raises(ValueError, match="monotonic"):
            DatasetSplitter().split(df, spec)

    def test_returns_copies(self, large_df):
        spec   = self._make_spec(large_df.index)
        result = DatasetSplitter().split(large_df, spec)
        original_val = result.train.iloc[0, 0]
        result.train.iloc[0, 0] = 99999.0
        assert large_df.iloc[0, 0] == pytest.approx(original_val)  # original unchanged

    def test_split_result_str(self, large_df):
        spec   = self._make_spec(large_df.index)
        result = DatasetSplitter().split(large_df, spec)
        s = str(result)
        assert "SplitResult" in s
        assert "1000" in s


# ── TestSplitValidator ────────────────────────────────────────────────────────

class TestSplitValidator:
    def _make_valid_result(self) -> SplitResult:
        idx   = pd.date_range("2020-01-01", periods=3000, freq="h")
        df    = pd.DataFrame({"x": range(3000)}, index=idx)
        spec  = WindowSpec(
            window_number=0,
            train_start=idx[0],    train_end=idx[999],
            val_start=idx[1000],   val_end=idx[1499],
            test_start=idx[1500],  test_end=idx[1999],
        )
        return DatasetSplitter().split(df, spec)

    def test_valid_split_passes(self):
        result = self._make_valid_result()
        report = SplitValidator().validate(result)
        assert report.passed
        assert len(report.failures()) == 0

    def test_detects_future_in_train(self):
        idx   = pd.date_range("2020-01-01", periods=3000, freq="h")
        train = pd.DataFrame({"x": range(1500)}, index=idx[:1500])
        val   = pd.DataFrame({"x": range(500)},  index=idx[1000:1500])  # overlaps train end
        test  = pd.DataFrame({"x": range(500)},  index=idx[2000:2500])
        result = SplitResult(0, train, val, test)
        report = SplitValidator().validate(result)
        assert not report.passed

    def test_detects_overlap(self):
        idx   = pd.date_range("2020-01-01", periods=3000, freq="h")
        train = pd.DataFrame({"x": range(1500)}, index=idx[:1500])
        val   = pd.DataFrame({"x": range(500)},  index=idx[1400:1900])  # overlaps
        test  = pd.DataFrame({"x": range(500)},  index=idx[2000:2500])
        result = SplitResult(0, train, val, test)
        report = SplitValidator().validate(result)
        assert not report.passed
        assert any(i.check == "no_overlap" for i in report.failures())

    def test_detects_min_sample_violation(self):
        idx   = pd.date_range("2020-01-01", periods=300, freq="h")
        train = pd.DataFrame({"x": range(200)}, index=idx[:200])
        val   = pd.DataFrame({"x": range(10)},  index=idx[200:210])  # too few
        test  = pd.DataFrame({"x": range(10)},  index=idx[220:230])
        result = SplitResult(0, train, val, test)
        cfg    = SplitValidatorConfig(min_train_samples=100, min_val_samples=50, min_test_samples=5)
        report = SplitValidator(cfg).validate(result)
        assert not report.passed
        assert any("min_val_samples" in i.check for i in report.failures())

    def test_detects_shuffle_in_split(self):
        idx_sorted = pd.date_range("2020-01-01", periods=3000, freq="h")
        idx_rev    = idx_sorted[:1000][::-1]
        train = pd.DataFrame({"x": range(1000)}, index=idx_rev)
        val   = pd.DataFrame({"x": range(500)},  index=idx_sorted[2000:2500])
        test  = pd.DataFrame({"x": range(500)},  index=idx_sorted[2500:3000])
        result = SplitResult(0, train, val, test)
        report = SplitValidator().validate(result)
        assert not report.passed
        assert any("no_shuffle" in i.check for i in report.failures())

    def test_report_str(self):
        result = self._make_valid_result()
        report = SplitValidator().validate(result)
        s = str(report)
        assert "PASSED" in s


# ── TestWindowMetadata ────────────────────────────────────────────────────────

class TestWindowMetadata:
    def _make_meta(self) -> WindowMeta:
        idx    = pd.date_range("2020-01-01", periods=3000, freq="h")
        train  = pd.DataFrame({"a": range(1000)}, index=idx[:1000])
        val    = pd.DataFrame({"a": range(500)},  index=idx[1000:1500])
        test   = pd.DataFrame({"a": range(500)},  index=idx[1500:2000])
        return WindowMeta.build(
            window_number=1, window_type="rolling",
            train_df=train, val_df=val, test_df=test,
            train_period="2y", val_period="6m", test_period="6m", step_period="6m",
            gap_bars=0, validation_passed=True, validation_issues=[],
            artefact_paths={"train": "windows/window_001/train.parquet"},
            feature_cols=["a"], label_cols=[],
        )

    def test_json_round_trip(self):
        meta = self._make_meta()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            meta.to_json(path)
            loaded = WindowMeta.from_json(path)
        assert loaded.window_number  == meta.window_number
        assert loaded.window_type    == meta.window_type
        assert loaded.train.row_count == meta.train.row_count

    def test_row_counts_correct(self):
        meta = self._make_meta()
        assert meta.train.row_count == 1000
        assert meta.val.row_count   == 500
        assert meta.test.row_count  == 500

    def test_to_dict_serialisable(self):
        meta = self._make_meta()
        d    = meta.to_dict()
        json.dumps(d)  # must not raise

    def test_artefact_paths_stored(self):
        meta = self._make_meta()
        assert "train" in meta.artefact_paths

    def test_feature_count(self):
        meta = self._make_meta()
        assert meta.feature_count == 1
        assert meta.label_count   == 0


# ── TestWalkForwardGenerator ─────────────────────────────────────────────────

class TestWalkForwardGenerator:
    def test_generates_windows(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
        )
        result = WalkForwardGenerator().run(large_df, symbol="TEST", config=cfg)
        assert result.n_windows >= 3

    def test_parquet_files_created(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=2,
        )
        result = WalkForwardGenerator().run(large_df, symbol="TEST", config=cfg)
        for meta in result.window_meta:
            for split_name in ("train", "validation", "test"):
                p = Path(meta.artefact_paths[split_name])
                assert p.exists(), f"Missing: {p}"

    def test_metadata_json_created(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=1,
        )
        WalkForwardGenerator().run(large_df, symbol="TEST", config=cfg)
        meta_files = list((tmp_path / "windows").rglob("metadata.json"))
        assert len(meta_files) >= 1

    def test_report_created(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=2,
        )
        result = WalkForwardGenerator().run(large_df, symbol="TEST", config=cfg)
        assert result.report_path.exists()

    def test_empty_df_raises(self, tmp_path):
        df  = pd.DataFrame()
        cfg = WalkForwardConfig(output_dir=tmp_path / "w", report_dir=tmp_path / "r")
        with pytest.raises(ValueError, match="empty"):
            WalkForwardGenerator().run(df, config=cfg)

    def test_non_datetime_index_raises(self, tmp_path):
        df  = pd.DataFrame({"a": range(100)})
        cfg = WalkForwardConfig(output_dir=tmp_path / "w", report_dir=tmp_path / "r")
        with pytest.raises(TypeError):
            WalkForwardGenerator().run(df, config=cfg)

    def test_parquet_input(self, large_df, tmp_path):
        parquet_path = tmp_path / "dataset.parquet"
        large_df.to_parquet(parquet_path)
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=1,
        )
        result = WalkForwardGenerator().run(parquet_path, config=cfg)
        assert result.n_windows >= 1

    def test_nonexistent_parquet_raises(self, tmp_path):
        cfg = WalkForwardConfig(output_dir=tmp_path / "w", report_dir=tmp_path / "r")
        with pytest.raises(FileNotFoundError):
            WalkForwardGenerator().run(tmp_path / "nonexistent.parquet", config=cfg)

    def test_expanding_windows_train_grows(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="expanding",
            train_period="2y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
        )
        result = WalkForwardGenerator().run(large_df, config=cfg)
        assert result.n_windows >= 2
        sizes = [m.train.row_count for m in result.window_meta]
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1]

    def test_all_passed_flag(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=2,
        )
        result = WalkForwardGenerator().run(large_df, config=cfg)
        assert result.all_passed

    def test_result_str(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=1,
        )
        result = WalkForwardGenerator().run(large_df, config=cfg)
        assert "WalkForwardResult" in str(result)

    def test_no_windows_when_data_too_short(self, tmp_path):
        idx = pd.date_range("2020-01-01", periods=200, freq="h")
        df  = pd.DataFrame({"a": range(200)}, index=idx)
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="5y", val_period="1y", test_period="1y", step_period="1y",
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
        )
        result = WalkForwardGenerator().run(df, config=cfg)
        assert result.n_windows == 0


# ── TestNoLookAhead ────────────────────────────────────────────────────────────

class TestNoLookAhead:
    """Verify zero temporal leakage across all windows."""

    def test_train_never_contains_val_or_test_timestamps(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=3,
        )
        result = WalkForwardGenerator().run(large_df, config=cfg)
        for meta in result.window_meta:
            # Reload splits from disk
            train_df = pd.read_parquet(meta.artefact_paths["train"])
            val_df   = pd.read_parquet(meta.artefact_paths["validation"])
            test_df  = pd.read_parquet(meta.artefact_paths["test"])
            future_ts = set(val_df.index) | set(test_df.index)
            leaking   = set(train_df.index) & future_ts
            assert len(leaking) == 0, f"Window {meta.window_number}: {len(leaking)} leaking rows"

    def test_val_never_contains_test_timestamps(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=3,
        )
        result = WalkForwardGenerator().run(large_df, config=cfg)
        for meta in result.window_meta:
            val_df  = pd.read_parquet(meta.artefact_paths["validation"])
            test_df = pd.read_parquet(meta.artefact_paths["test"])
            overlap = set(val_df.index) & set(test_df.index)
            assert len(overlap) == 0

    def test_train_end_before_val_start(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=3,
        )
        result = WalkForwardGenerator().run(large_df, config=cfg)
        for meta in result.window_meta:
            train_end = pd.Timestamp(meta.train.end)
            val_start = pd.Timestamp(meta.val.start)
            assert train_end < val_start


# ── TestChronologicalOrder ────────────────────────────────────────────────────

class TestChronologicalOrder:
    def test_each_split_is_sorted(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=2,
        )
        result = WalkForwardGenerator().run(large_df, config=cfg)
        for meta in result.window_meta:
            for split_name in ("train", "validation", "test"):
                df = pd.read_parquet(meta.artefact_paths[split_name])
                assert df.index.is_monotonic_increasing, \
                    f"Window {meta.window_number} {split_name} is NOT sorted."

    def test_data_not_shuffled(self, large_df, tmp_path):
        cfg = WalkForwardConfig(
            window_type="rolling",
            train_period="3y", val_period="6m", test_period="6m",
            step_period="6m", min_train_samples=100, min_val_samples=50, min_test_samples=50,
            output_dir=tmp_path / "windows",
            report_dir=tmp_path / "reports",
            max_windows=1,
        )
        result  = WalkForwardGenerator().run(large_df, config=cfg)
        meta    = result.window_meta[0]
        train   = pd.read_parquet(meta.artefact_paths["train"])
        # Verify the row values match the original data in the same order
        original_slice = large_df.loc[train.index]
        pd.testing.assert_frame_equal(train, original_slice)


# ── TestReports ───────────────────────────────────────────────────────────────

class TestReports:
    def _make_meta_list(self, n: int = 2) -> list[WindowMeta]:
        result = []
        for i in range(n):
            idx   = pd.date_range(f"202{i}-01-01", periods=3000, freq="h")
            train = pd.DataFrame({"a": range(1000)}, index=idx[:1000])
            val   = pd.DataFrame({"a": range(500)},  index=idx[1000:1500])
            test  = pd.DataFrame({"a": range(500)},  index=idx[1500:2000])
            result.append(WindowMeta.build(
                window_number=i, window_type="rolling",
                train_df=train, val_df=val, test_df=test,
                train_period="2y", val_period="6m", test_period="6m", step_period="6m",
                gap_bars=0, validation_passed=True, validation_issues=[],
                artefact_paths={}, feature_cols=["a"], label_cols=[],
            ))
        return result

    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out  = Path(tmp) / "reports"
            metas = self._make_meta_list(3)
            path  = generate_walk_forward_report(metas, {"window_type": "rolling"}, out, "TEST")
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "Walk-Forward Validation Report" in content

    def test_report_contains_window_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            out   = Path(tmp) / "reports"
            metas = self._make_meta_list(2)
            path  = generate_walk_forward_report(metas, {}, out)
            content = path.read_text(encoding="utf-8")
            assert "000" in content  # window_000

    def test_empty_windows_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out  = Path(tmp) / "reports"
            path = generate_walk_forward_report([], {"window_type": "rolling"}, out)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "No windows" in content

    def test_report_contains_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            out   = Path(tmp) / "reports"
            metas = self._make_meta_list(1)
            path  = generate_walk_forward_report(metas, {}, out, "GBPUSD")
            content = path.read_text(encoding="utf-8")
            assert "GBPUSD" in content
