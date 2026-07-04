"""
Tests for src/dataset — Dataset Builder.

All tests use synthetic in-memory DataFrames and temporary directories.
No Feature Store or Label Store I/O is required — tests run fully offline.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_features(n: int = 300, n_feats: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic feature DataFrame with DatetimeIndex."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2022-01-01", periods=n, freq="h")
    cols = (
        [f"h1_sma_{i}" for i in range(5)]
        + [f"h1_rsi_{i}" for i in range(5)]
        + [f"h4_atr_{i}" for i in range(5)]
        + [f"d1_vol_{i}" for i in range(5)]
    )[:n_feats]
    data = rng.standard_normal((n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def _make_labels(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Synthetic label DataFrame; last 50 rows are NaN (forward-looking tail)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    data: dict = {
        "direction_1b":   rng.choice([0.0, 1.0, 2.0], size=n).astype(float),
        "bias_1b":        rng.choice([0.0, 1.0],        size=n).astype(float),
        "fwd_return_1b":  rng.standard_normal(n) * 0.001,
        "setup_quality":  rng.choice([0.0, 1.0, 2.0, 3.0], size=n).astype(float),
        "setup_score":    rng.uniform(0, 100, n),
        "entry_signal":   rng.choice([0.0, 1.0, 2.0], size=n).astype(float),
        "long_outcome":   rng.choice([0.0, 1.0, 2.0], size=n).astype(float),
        "outcome":        rng.choice([0.0, 1.0, 2.0], size=n).astype(float),
        "mfe_pct":        rng.uniform(0, 0.02, n),
        "mgmt_strategy":  rng.choice([0.0, 1.0, 2.0, 3.0], size=n).astype(float),
    }
    df = pd.DataFrame(data, index=idx)
    df.iloc[-50:] = np.nan   # simulate forward-looking NaN tail
    return df


def _make_config(
    tmp_path: Path,
    feature_set: str = "all",
    label_groups=None,
    primary_target: str = "direction_1b",
    drop_na: bool = True,
    validate: bool = True,
    formats: list = None,
) -> "DatasetConfig":
    from src.dataset import DatasetConfig
    return DatasetConfig(
        symbol="TESTSYM",
        feature_set=feature_set,
        label_groups=label_groups,
        primary_target=primary_target,
        drop_na_labels=drop_na,
        output_formats=formats or ["parquet"],
        output_dir=tmp_path / "ml",
        validate=validate,
        min_rows=10,
    )


def _builder(tmp_path: Path) -> "DatasetBuilder":
    from src.dataset import DatasetBuilder
    return DatasetBuilder(
        output_dir=tmp_path / "ml",
        report_dir=tmp_path / "reports",
    )


# ── DatasetValidator ─────────────────────────────────────────────────────────

class TestDatasetValidator:
    def _validator(self):
        from src.dataset import DatasetValidator, DatasetValidatorConfig
        return DatasetValidator(DatasetValidatorConfig(min_rows=10))

    def _valid_dataset(self, n=200):
        feats = _make_features(n)
        labels = _make_labels(n)
        labels.iloc[-50:] = np.nan
        ds = feats.join(labels).dropna(subset=["direction_1b"])
        feat_cols = list(feats.columns)
        lbl_cols  = list(labels.columns)
        return ds, feat_cols, lbl_cols

    def test_passes_on_clean_data(self):
        ds, fc, lc = self._valid_dataset()
        rep = self._validator().validate(ds, fc, lc)
        assert not rep.failures(), [str(i) for i in rep.failures()]

    def test_fails_on_empty_dataset(self):
        empty = pd.DataFrame()
        rep = self._validator().validate(empty, [], [])
        assert not rep.passed

    def test_detects_non_monotonic_index(self):
        ds, fc, lc = self._valid_dataset()
        shuffled = ds.sample(frac=1, random_state=0)
        rep = self._validator().validate(shuffled, fc, lc)
        fail_checks = [i.check for i in rep.failures()]
        assert "time_ordering" in fail_checks

    def test_detects_duplicate_timestamps(self):
        ds, fc, lc = self._valid_dataset()
        ds_dup = pd.concat([ds, ds.iloc[:5]])
        rep = self._validator().validate(ds_dup, fc, lc)
        fail_checks = [i.check for i in rep.failures()]
        assert "duplicate_rows" in fail_checks

    def test_detects_missing_target(self):
        ds, fc, lc = self._valid_dataset()
        rep = self._validator().validate(ds, fc, lc, primary_target="nonexistent_col")
        fail_checks = [i.check for i in rep.failures()]
        assert "target_available" in fail_checks

    def test_target_with_all_nan_fails(self):
        ds, fc, lc = self._valid_dataset()
        ds = ds.copy()
        ds["direction_1b"] = np.nan
        rep = self._validator().validate(ds, fc, lc, primary_target="direction_1b")
        fail_checks = [i.check for i in rep.failures()]
        assert "target_available" in fail_checks

    def test_detects_duplicate_columns(self):
        ds, fc, lc = self._valid_dataset()
        ds_dup = pd.concat([ds, ds[["direction_1b"]]], axis=1)
        rep = self._validator().validate(ds_dup, fc, lc)
        fail_checks = [i.check for i in rep.failures()]
        assert "duplicate_cols" in fail_checks

    def test_expected_columns_check(self):
        from src.dataset import DatasetValidator, DatasetValidatorConfig
        ds, fc, lc = self._valid_dataset()
        val = DatasetValidator(DatasetValidatorConfig(
            min_rows=10,
            expected_columns=["nonexistent_feature_xyz"]
        ))
        rep = val.validate(ds, fc, lc)
        fail_checks = [i.check for i in rep.failures()]
        assert "schema_columns" in fail_checks

    def test_min_rows_enforced(self):
        from src.dataset import DatasetValidator, DatasetValidatorConfig
        ds, fc, lc = self._valid_dataset()
        val = DatasetValidator(DatasetValidatorConfig(min_rows=99999))
        rep = val.validate(ds, fc, lc)
        fail_checks = [i.check for i in rep.failures()]
        assert "min_rows" in fail_checks


# ── DatasetMetadata ───────────────────────────────────────────────────────────

class TestDatasetMetadata:
    def _build(self) -> "DatasetMeta":
        from src.dataset import DatasetMeta
        feats  = _make_features(200)
        labels = _make_labels(200)
        ds = feats.join(labels).dropna(subset=["direction_1b"])
        return DatasetMeta.build(
            dataset=ds,
            feature_columns=list(feats.columns),
            label_columns=list(labels.columns),
            symbol="EURUSD",
            feature_set="top50",
            label_groups=["market_bias"],
        )

    def test_basic_fields(self):
        meta = self._build()
        assert meta.symbol == "EURUSD"
        assert meta.feature_count == 20
        assert meta.row_count > 0

    def test_json_roundtrip(self, tmp_path):
        from src.dataset import DatasetMeta
        meta  = self._build()
        path  = tmp_path / "meta.json"
        meta.to_json(path)
        meta2 = DatasetMeta.from_json(path)
        assert meta2.symbol       == meta.symbol
        assert meta2.feature_count == meta.feature_count
        assert meta2.row_count    == meta.row_count

    def test_column_summaries_populated(self):
        meta = self._build()
        assert len(meta.column_summaries) > 0
        for cs in meta.column_summaries:
            assert 0.0 <= cs.nan_rate <= 1.0

    def test_missing_rate_in_unit_interval(self):
        meta = self._build()
        assert 0.0 <= meta.missing_rate <= 1.0

    def test_start_end_dates_populated(self):
        meta = self._build()
        assert meta.start_date != ""
        assert meta.end_date   != ""


# ── DatasetLoader ─────────────────────────────────────────────────────────────

class TestDatasetLoader:
    def _loader(self, tmp_path: Path) -> "DatasetLoader":
        from src.dataset import DatasetLoader
        return DatasetLoader(
            feature_store_dir=tmp_path / "fs",
            schema_dir=tmp_path / "schema",
            label_dir=tmp_path / "labels",
            feature_quality_dir=tmp_path / "fq",
        )

    def test_load_features_all(self, tmp_path):
        feats = _make_features()
        path  = tmp_path / "feat.parquet"
        feats.to_parquet(path)
        loader = self._loader(tmp_path)
        loaded = loader.load_features("EURUSD", feature_set="all", parquet_path=path)
        assert loaded.shape == feats.shape

    def test_load_features_custom(self, tmp_path):
        feats  = _make_features()
        path   = tmp_path / "feat.parquet"
        feats.to_parquet(path)
        loader = self._loader(tmp_path)
        subset = list(feats.columns[:5])
        loaded = loader.load_features(
            "EURUSD", feature_set="custom",
            custom_features=subset, parquet_path=path,
        )
        assert list(loaded.columns) == subset

    def test_load_features_top25_from_json(self, tmp_path):
        feats   = _make_features(n_feats=20)
        path    = tmp_path / "feat.parquet"
        feats.to_parquet(path)
        # Write a feature-quality JSON
        fq_dir  = tmp_path / "fq"
        fq_dir.mkdir()
        top25   = list(feats.columns[:5])
        (fq_dir / "selected_features_top25.json").write_text(
            json.dumps(top25), encoding="utf-8"
        )
        loader = self._loader(tmp_path)
        loaded = loader.load_features("EURUSD", feature_set="top25", parquet_path=path)
        assert list(loaded.columns) == top25

    def test_get_feature_list_handles_dict_format(self, tmp_path):
        fq_dir = tmp_path / "fq"
        fq_dir.mkdir()
        (fq_dir / "selected_features_top50.json").write_text(
            json.dumps({"features": ["a", "b", "c"]}), encoding="utf-8"
        )
        loader = self._loader(tmp_path)
        lst = loader.get_feature_list("top50")
        assert lst == ["a", "b", "c"]

    def test_load_labels_all_groups(self, tmp_path):
        labels = _make_labels()
        sym_dir = tmp_path / "labels" / "EURUSD"
        sym_dir.mkdir(parents=True)
        lbl_path = sym_dir / "labels_EURUSD_v1.parquet"
        labels.to_parquet(lbl_path)
        loader = self._loader(tmp_path)
        loaded = loader.load_labels("EURUSD", label_groups=None)
        assert loaded.shape == labels.shape

    def test_load_labels_market_bias_group(self, tmp_path):
        labels   = _make_labels()
        sym_dir  = tmp_path / "labels" / "EURUSD"
        sym_dir.mkdir(parents=True)
        lbl_path = sym_dir / "labels_EURUSD_v1.parquet"
        labels.to_parquet(lbl_path)
        loader = self._loader(tmp_path)
        loaded = loader.load_labels("EURUSD", label_groups=["market_bias"])
        bias_cols = [c for c in labels.columns
                     if any(c.startswith(p)
                            for p in ("fwd_return_","direction_","bias_","confidence_","probability_"))]
        for col in bias_cols:
            assert col in loaded.columns

    def test_load_labels_direct_parquet(self, tmp_path):
        labels = _make_labels()
        path   = tmp_path / "lbl.parquet"
        labels.to_parquet(path)
        loader = self._loader(tmp_path)
        loaded = loader.load_labels("X", parquet_path=path)
        assert loaded.shape == labels.shape

    def test_invalid_label_group_raises(self, tmp_path):
        labels = _make_labels()
        path   = tmp_path / "lbl.parquet"
        labels.to_parquet(path)
        loader = self._loader(tmp_path)
        with pytest.raises(ValueError, match="Unknown label group"):
            loader.load_labels("X", label_groups=["nonexistent_group"], parquet_path=path)

    def test_list_label_versions(self, tmp_path):
        sym_dir = tmp_path / "labels" / "EURUSD"
        sym_dir.mkdir(parents=True)
        for v in [1, 2, 3]:
            (sym_dir / f"labels_EURUSD_v{v}.parquet").touch()
        loader   = self._loader(tmp_path)
        versions = loader.list_label_versions("EURUSD")
        assert versions == [1, 2, 3]

    def test_missing_label_dir_raises(self, tmp_path):
        loader = self._loader(tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load_labels("NONEXISTENT_SYMBOL")


# ── DatasetBuilder ────────────────────────────────────────────────────────────

class TestDatasetBuilder:
    def test_build_from_dataframes_produces_dataset(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert isinstance(result.dataset, pd.DataFrame)
        assert len(result.dataset) > 0

    def test_features_and_labels_both_present(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert result.n_features > 0
        assert result.n_labels   > 0

    def test_nan_label_rows_dropped(self, tmp_path):
        feats  = _make_features(300)
        labels = _make_labels(300)        # last 50 rows are NaN
        cfg    = _make_config(tmp_path, primary_target="direction_1b", drop_na=True)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert result.dataset["direction_1b"].isna().sum() == 0

    def test_no_drop_na_keeps_nans(self, tmp_path):
        feats  = _make_features(300)
        labels = _make_labels(300)
        cfg    = _make_config(tmp_path, drop_na=False)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        # NaN tail should still exist
        assert result.dataset["direction_1b"].isna().sum() > 0

    def test_parquet_saved(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert result.parquet_path is not None
        assert result.parquet_path.exists()

    def test_csv_saved_when_requested(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path, formats=["parquet", "csv"])
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert result.csv_path is not None
        assert result.csv_path.exists()

    def test_parquet_loadable(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        loaded = pd.read_parquet(result.parquet_path)
        assert loaded.shape == result.dataset.shape

    def test_versioning_increments(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        r1 = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        r2 = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert r1.parquet_path != r2.parquet_path
        assert "v1" in r1.parquet_path.name
        assert "v2" in r2.parquet_path.name

    def test_input_not_mutated(self, tmp_path):
        feats  = _make_features().copy()
        labels = _make_labels().copy()
        feat_copy  = feats.copy()
        label_copy = labels.copy()
        cfg    = _make_config(tmp_path)
        _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        pd.testing.assert_frame_equal(feats,  feat_copy)
        pd.testing.assert_frame_equal(labels, label_copy)

    def test_time_ordering_preserved(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert result.dataset.index.is_monotonic_increasing

    def test_label_cols_not_in_feature_cols(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        feat_set  = set(result.feature_columns)
        label_set = set(result.label_columns)
        assert feat_set.isdisjoint(label_set), \
            f"Overlap: {feat_set & label_set}"

    def test_feature_set_custom(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        subset = list(feats.columns[:3])
        from src.dataset import DatasetConfig
        cfg = DatasetConfig(
            symbol="TESTSYM",
            feature_set="custom",
            custom_features=subset,
            output_dir=tmp_path / "ml",
            min_rows=10,
        )
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert result.feature_columns == subset

    def test_label_group_selection(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path, label_groups=["market_bias"])
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        for col in result.label_columns:
            assert any(col.startswith(p)
                       for p in ("fwd_return_","direction_","bias_","confidence_","probability_"))

    def test_metadata_generated(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert result.metadata.symbol == "TESTSYM"
        assert result.metadata.row_count == result.n_rows

    def test_validation_report_generated(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path, validate=True)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert result.validation is not None
        assert isinstance(result.validation.passed, bool)

    def test_reports_written(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert (tmp_path / "reports" / "training_dataset_report.md").exists()
        assert (tmp_path / "reports" / "training_dataset_metadata.json").exists()

    def test_build_via_parquet_paths(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        fp     = tmp_path / "feat.parquet";   feats.to_parquet(fp)
        lp     = tmp_path / "label.parquet"; labels.to_parquet(lp)
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build(cfg, feature_parquet=fp, label_parquet=lp)
        assert len(result.dataset) > 0


# ── Report Generator ──────────────────────────────────────────────────────────

class TestDatasetReports:
    def test_markdown_report_written(self, tmp_path):
        from src.dataset import DatasetReportGenerator, DatasetMeta
        feats  = _make_features(200)
        labels = _make_labels(200)
        ds     = feats.join(labels).dropna(subset=["direction_1b"])
        meta   = DatasetMeta.build(
            dataset=ds,
            feature_columns=list(feats.columns),
            label_columns=list(labels.columns),
            symbol="EURUSD",
        )
        reporter = DatasetReportGenerator(tmp_path / "reports")
        paths = reporter.generate_all(ds, meta)
        assert paths["report"].exists()
        content = paths["report"].read_text()
        assert "EURUSD" in content
        assert "Feature Summary" in content
        assert "Label Summary" in content

    def test_metadata_json_written(self, tmp_path):
        from src.dataset import DatasetReportGenerator, DatasetMeta
        feats  = _make_features(100)
        labels = _make_labels(100)
        ds     = feats.join(labels).dropna(subset=["direction_1b"])
        meta   = DatasetMeta.build(
            dataset=ds, feature_columns=list(feats.columns),
            label_columns=list(labels.columns), symbol="GBPUSD",
        )
        reporter = DatasetReportGenerator(tmp_path / "reports")
        paths    = reporter.generate_all(ds, meta)
        assert paths["metadata"].exists()
        loaded = json.loads(paths["metadata"].read_text())
        assert loaded["symbol"] == "GBPUSD"


# ── No Look-Ahead Invariants ──────────────────────────────────────────────────

class TestNoLookAhead:
    def test_index_matches_original_feature_index(self, tmp_path):
        feats  = _make_features(300)
        labels = _make_labels(300)
        cfg    = _make_config(tmp_path, drop_na=True)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        # Every timestamp in result must have been in the original features
        assert set(result.dataset.index).issubset(set(feats.index))

    def test_no_temporal_shift_applied(self, tmp_path):
        feats  = _make_features(300)
        labels = _make_labels(300)
        cfg    = _make_config(tmp_path, drop_na=False)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        # Feature row at time T and label row at time T must share the same index
        ds      = result.dataset
        feat_at_t   = feats.loc[ds.index, list(feats.columns)[:1]]
        result_feat = ds[list(feats.columns)[:1]]
        pd.testing.assert_frame_equal(feat_at_t, result_feat)

    def test_label_columns_never_overlap_feature_columns(self, tmp_path):
        feats  = _make_features()
        labels = _make_labels()
        cfg    = _make_config(tmp_path)
        result = _builder(tmp_path).build_from_dataframes(feats, labels, cfg)
        assert set(result.feature_columns).isdisjoint(set(result.label_columns))
