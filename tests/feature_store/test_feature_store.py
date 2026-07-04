"""
Comprehensive tests for the Feature Store.

Coverage
--------
* Unit: prefix utilities, schema versioning, dataset hash, schema model
* Schema: registry, manager, validator, compatibility checker, contracts
* Catalog: build, search, serialisation
* Manifest: generate, save/load, index
* Loader: save, load, versioning, archive
* Store: save, load, load_* variants, validate, catalog, introspection
* Integration: full round-trip (save → load → validate)
* Performance: save/load 100 k rows in < 5 s
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Feature store imports ──────────────────────────────────────────────────────
from src.feature_store import (
    CompatibilityChecker,
    DatasetManifest,
    DatasetSchema,
    FeatureCatalog,
    FeatureContract,
    FeatureLoader,
    FeatureSchema,
    FeatureStore,
    ManifestIndex,
    SchemaContract,
    SchemaManager,
    SchemaRegistry,
    SchemaValidator,
    SemanticVersion,
    ValidationReport,
    VALID_PREFIXES,
    compute_features_hash,
    compute_schema_hash,
    determine_version_bump,
    extract_prefix,
    group_by_prefix,
    is_valid_prefix,
    validate_prefix,
)
from src.feature_store.exceptions import (
    CatalogError,
    CompatibilityError,
    DatasetNotFoundError,
    HashVerificationError,
    InvalidPrefixError,
    ManifestError,
    SchemaFrozenError,
    SchemaNotFoundError,
    SchemaMismatchError,
    ValidationError,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

SYMBOL = "EURUSD"


def _make_index(n: int = 10) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")


def _make_df(n: int = 10, cols: list[str] | None = None) -> pd.DataFrame:
    idx  = _make_index(n)
    cols = cols or ["h1_rsi_14", "m15_atr_14", "daily_trend", "tech_macd"]
    rng  = np.random.default_rng(42)
    return pd.DataFrame(
        {c: rng.random(n).astype("float64") for c in cols},
        index=idx,
    )


def _make_feature(name: str, category: str = "technical") -> FeatureSchema:
    return FeatureSchema(
        name=name,
        prefix=extract_prefix(name),
        category=category,
        description=f"Feature {name}",
        source_module="test",
        timeframe="",
        dependencies=[],
        dtype="float64",
        nullable=False,
        default_value=None,
        min_value=None,
        max_value=None,
        units="",
        example_value=None,
        version="1.0.0",
        author="tester",
        created_date="2024-01-01",
        updated_date="2024-01-01",
        deprecated=False,
        tags=[],
    )


def _make_schema(
    symbol: str = SYMBOL,
    version: str = "1.0.0",
    cols: list[str] | None = None,
) -> DatasetSchema:
    cols = cols or ["h1_rsi_14", "m15_atr_14", "daily_trend", "tech_macd"]
    features = {c: _make_feature(c) for c in cols}
    return DatasetSchema(
        version=version,
        symbol=symbol,
        features=features,
        created_at="2024-01-01T00:00:00+00:00",
        frozen=False,
        schema_hash=compute_features_hash(features),
    )


# =============================================================================
# 1. Prefix utilities
# =============================================================================


class TestValidPrefixes:
    def test_all_prefixes_present(self):
        expected = {
            "weekly_", "daily_", "h4_", "h1_", "m15_",
            "ms_", "liq_", "tech_", "stat_", "vol_",
            "sess_", "label_", "future_", "macro_", "news_", "sent_",
        }
        assert expected.issubset(VALID_PREFIXES)

    def test_is_valid_prefix_true(self):
        assert is_valid_prefix("h1_rsi")
        assert is_valid_prefix("weekly_high")
        assert is_valid_prefix("tech_macd")

    def test_is_valid_prefix_false(self):
        assert not is_valid_prefix("unknown_feat")
        assert not is_valid_prefix("close")

    def test_extract_prefix_longest_match(self):
        assert extract_prefix("weekly_high") == "weekly_"
        assert extract_prefix("m15_ema_20")  == "m15_"
        assert extract_prefix("daily_close") == "daily_"
        assert extract_prefix("no_match")    == ""

    def test_validate_prefix_raises(self):
        with pytest.raises(InvalidPrefixError):
            validate_prefix("bad_feature")

    def test_validate_prefix_ok(self):
        validate_prefix("h4_ema_50")  # should not raise

    def test_group_by_prefix(self):
        names = ["h1_rsi", "h1_macd", "m15_ema", "tech_bband"]
        grouped = group_by_prefix(names)
        assert "h1_"   in grouped
        assert "m15_"  in grouped
        assert "tech_" in grouped
        assert len(grouped["h1_"]) == 2


# =============================================================================
# 2. Semantic versioning
# =============================================================================


class TestSemanticVersion:
    def test_parse_valid(self):
        v = SemanticVersion.parse("2.3.1")
        assert v.major == 2
        assert v.minor == 3
        assert v.patch == 1

    def test_str(self):
        assert str(SemanticVersion(1, 2, 3)) == "1.2.3"

    def test_compatible_same_major(self):
        a = SemanticVersion.parse("1.0.0")
        b = SemanticVersion.parse("1.5.2")
        assert a.is_compatible_with(b)

    def test_incompatible_different_major(self):
        a = SemanticVersion.parse("1.0.0")
        b = SemanticVersion.parse("2.0.0")
        assert not a.is_compatible_with(b)

    def test_bumped_major(self):
        v = SemanticVersion.parse("1.2.3").bumped_major()
        assert v == SemanticVersion(2, 0, 0)

    def test_bumped_minor(self):
        v = SemanticVersion.parse("1.2.3").bumped_minor()
        assert v == SemanticVersion(1, 3, 0)

    def test_bumped_patch(self):
        v = SemanticVersion.parse("1.2.3").bumped_patch()
        assert v == SemanticVersion(1, 2, 4)


class TestVersionBump:
    def _feats(self, names):
        return {n: _make_feature(n) for n in names}

    def test_major_on_removal(self):
        old = self._feats(["h1_a", "h1_b"])
        new = self._feats(["h1_a"])
        assert determine_version_bump(old, new) == "major"

    def test_minor_on_addition(self):
        old = self._feats(["h1_a"])
        new = self._feats(["h1_a", "h1_b"])
        assert determine_version_bump(old, new) == "minor"

    def test_none_on_identical(self):
        old = self._feats(["h1_a"])
        assert determine_version_bump(old, old) == "none"


# =============================================================================
# 3. Dataset hash
# =============================================================================


class TestDatasetHash:
    def test_deterministic(self):
        feats = {"h1_a": _make_feature("h1_a")}
        h1 = compute_features_hash(feats)
        h2 = compute_features_hash(feats)
        assert h1 == h2

    def test_different_on_name_change(self):
        f1 = {"h1_a": _make_feature("h1_a")}
        f2 = {"h1_b": _make_feature("h1_b")}
        assert compute_features_hash(f1) != compute_features_hash(f2)

    def test_different_on_dtype_change(self):
        f1 = {"h1_a": _make_feature("h1_a")}
        f2 = dict(f1)
        fa = _make_feature("h1_a")
        fa = FeatureSchema(
            **{**fa.__dict__, "dtype": "float32"}
        )
        f2["h1_a"] = fa
        assert compute_features_hash(f1) != compute_features_hash(f2)

    def test_hex_length(self):
        h = compute_features_hash({"h1_a": _make_feature("h1_a")})
        assert len(h) == 64   # SHA-256 hex

    def test_compute_schema_hash(self):
        schema = _make_schema()
        h = compute_schema_hash(schema)
        assert isinstance(h, str) and len(h) == 64


# =============================================================================
# 4. FeatureSchema / DatasetSchema model
# =============================================================================


class TestFeatureSchemaModel:
    def test_to_dict_roundtrip(self):
        fs = _make_feature("tech_macd")
        assert FeatureSchema.from_dict(fs.to_dict()).name == "tech_macd"

    def test_structural_tuple_stable(self):
        fs = _make_feature("h1_rsi_14")
        t1 = fs.structural_tuple()
        t2 = fs.structural_tuple()
        assert t1 == t2


class TestDatasetSchemaModel:
    def test_feature_names(self):
        schema = _make_schema(cols=["h1_a", "m15_b"])
        assert schema.feature_names == ["h1_a", "m15_b"]

    def test_prefixes(self):
        schema = _make_schema(cols=["h1_a", "m15_b"])
        assert "h1_" in schema.prefixes
        assert "m15_" in schema.prefixes

    def test_feature_count(self):
        schema = _make_schema(cols=["h1_a", "h1_b", "m15_c"])
        assert schema.feature_count == 3

    def test_to_dict_roundtrip(self, tmp_path):
        schema = _make_schema()
        d      = schema.to_dict()
        schema2 = DatasetSchema.from_dict(d)
        assert schema2.version == schema.version
        assert set(schema2.features) == set(schema.features)

    def test_to_json_roundtrip(self, tmp_path):
        schema = _make_schema()
        path   = tmp_path / "schema.json"
        schema.to_json(path)
        schema2 = DatasetSchema.from_json(path)
        assert schema2.schema_hash == schema.schema_hash


# =============================================================================
# 5. SchemaRegistry
# =============================================================================


class TestSchemaRegistry:
    def test_register_and_get(self, tmp_path):
        reg    = SchemaRegistry(tmp_path)
        schema = _make_schema()
        reg.register(schema)
        loaded = reg.get(SYMBOL, "1.0.0")
        assert loaded.version == "1.0.0"

    def test_get_latest(self, tmp_path):
        reg = SchemaRegistry(tmp_path)
        for ver in ["1.0.0", "1.1.0", "2.0.0"]:
            s = _make_schema(version=ver)
            reg.register(s)
        latest = reg.get_latest(SYMBOL)
        assert latest.version == "2.0.0"

    def test_list_versions(self, tmp_path):
        reg = SchemaRegistry(tmp_path)
        for ver in ["1.0.0", "1.1.0"]:
            reg.register(_make_schema(version=ver))
        assert reg.list_versions(SYMBOL) == ["1.0.0", "1.1.0"]

    def test_exists(self, tmp_path):
        reg = SchemaRegistry(tmp_path)
        assert not reg.exists(SYMBOL, "1.0.0")
        reg.register(_make_schema())
        assert reg.exists(SYMBOL, "1.0.0")

    def test_not_found_raises(self, tmp_path):
        reg = SchemaRegistry(tmp_path)
        with pytest.raises(SchemaNotFoundError):
            reg.get(SYMBOL, "9.9.9")

    def test_pin_overrides_latest(self, tmp_path):
        reg = SchemaRegistry(tmp_path)
        reg.register(_make_schema(version="1.0.0"))
        reg.register(_make_schema(version="1.1.0"))
        reg.pin(SYMBOL, "1.0.0")
        assert reg.get_latest(SYMBOL).version == "1.0.0"

    def test_unpin_restores_latest(self, tmp_path):
        reg = SchemaRegistry(tmp_path)
        reg.register(_make_schema(version="1.0.0"))
        reg.register(_make_schema(version="1.1.0"))
        reg.pin(SYMBOL, "1.0.0")
        reg.unpin(SYMBOL)
        assert reg.get_latest(SYMBOL).version == "1.1.0"

    def test_list_symbols(self, tmp_path):
        reg = SchemaRegistry(tmp_path)
        reg.register(_make_schema(symbol="EURUSD"))
        reg.register(_make_schema(symbol="GBPUSD"))
        assert set(reg.list_symbols()) == {"EURUSD", "GBPUSD"}


# =============================================================================
# 6. SchemaManager
# =============================================================================


class TestSchemaManager:
    def test_infer_from_dataframe(self, tmp_path):
        reg  = SchemaRegistry(tmp_path)
        mgr  = SchemaManager(reg)
        df   = _make_df()
        schema = mgr.infer_from_dataframe(df, SYMBOL)
        assert set(schema.features) == set(df.columns)
        assert schema.schema_hash

    def test_freeze_registers_schema(self, tmp_path):
        reg    = SchemaRegistry(tmp_path)
        mgr    = SchemaManager(reg)
        df     = _make_df()
        schema = mgr.infer_from_dataframe(df, SYMBOL)
        frozen = mgr.freeze(schema)
        assert frozen.frozen
        assert reg.exists(SYMBOL, frozen.version)

    def test_evolve_minor_bump(self, tmp_path):
        reg    = SchemaRegistry(tmp_path)
        mgr    = SchemaManager(reg)
        df     = _make_df()
        schema = mgr.infer_from_dataframe(df, SYMBOL)
        mgr.freeze(schema)
        new_feats = dict(schema.features)
        new_feats["h1_new_feat"] = _make_feature("h1_new_feat")
        evolved = mgr.evolve(schema, new_feats)
        assert SemanticVersion.parse(evolved.version).minor == 1

    def test_rollback(self, tmp_path):
        reg = SchemaRegistry(tmp_path)
        mgr = SchemaManager(reg)
        for ver in ["1.0.0", "1.1.0"]:
            reg.register(_make_schema(version=ver))
        mgr.rollback(SYMBOL, "1.0.0")
        assert reg.get_latest(SYMBOL).version == "1.0.0"

    def test_compare(self, tmp_path):
        reg  = SchemaRegistry(tmp_path)
        mgr  = SchemaManager(reg)
        s1   = _make_schema(cols=["h1_a", "h1_b"])
        s2   = _make_schema(cols=["h1_a", "h1_c"], version="1.1.0")
        diff = mgr.compare(s1, s2)
        assert "h1_c" in diff["added"]
        assert "h1_b" in diff["removed"]


# =============================================================================
# 7. SchemaValidator
# =============================================================================


class TestSchemaValidator:
    def test_valid_df_passes(self):
        schema = _make_schema()
        df     = _make_df()
        report = SchemaValidator().validate(df, schema)
        assert report.is_valid

    def test_missing_feature_fails(self):
        schema = _make_schema(cols=["h1_a", "h1_b"])
        df     = pd.DataFrame({"h1_a": [1.0]})
        report = SchemaValidator().validate(df, schema, strict=True)
        assert not report.is_valid
        assert any("h1_b" in e for e in report.errors)

    def test_dtype_mismatch_fails(self):
        schema = _make_schema(cols=["h1_a"])
        df     = pd.DataFrame({"h1_a": ["bad_string"]})
        report = SchemaValidator().validate(df, schema, strict=True)
        assert not report.is_valid


# =============================================================================
# 8. CompatibilityChecker
# =============================================================================


class TestCompatibilityChecker:
    def test_identical_schemas_compatible(self):
        s = _make_schema()
        r = CompatibilityChecker().check(s, s)
        assert r.is_compatible

    def test_added_feature_compatible(self):
        s1 = _make_schema(cols=["h1_a"])
        s2 = _make_schema(cols=["h1_a", "h1_b"], version="1.1.0")
        r  = CompatibilityChecker().check(s1, s2)
        assert r.is_compatible
        assert "h1_b" in r.new_features

    def test_removed_feature_incompatible(self):
        s1 = _make_schema(cols=["h1_a", "h1_b"])
        s2 = _make_schema(cols=["h1_a"], version="2.0.0")
        r  = CompatibilityChecker().check(s1, s2)
        assert not r.is_compatible
        assert "h1_b" in r.removed_features

    def test_migration_plan(self):
        s1 = _make_schema(cols=["h1_a", "h1_b"])
        s2 = _make_schema(cols=["h1_a", "h1_c"], version="2.0.0")
        plan = CompatibilityChecker().get_migration_plan(s1, s2)
        actions = {a["action"] for a in plan["actions"]}
        assert "remove_column" in actions
        assert "add_column" in actions


# =============================================================================
# 9. Feature Catalog
# =============================================================================


class TestFeatureCatalog:
    def test_build_counts(self):
        schema  = _make_schema(cols=["h1_a", "m15_b", "tech_c"])
        catalog = FeatureCatalog.from_schema(schema)
        assert catalog.feature_count == 3

    def test_search_by_prefix(self):
        schema  = _make_schema(cols=["h1_a", "m15_b", "h1_c"])
        catalog = FeatureCatalog.from_schema(schema)
        results = catalog.search(prefix="h1_")
        assert all(f.prefix == "h1_" for f in results)
        assert len(results) == 2

    def test_search_by_category(self):
        cols    = ["h1_a", "m15_b"]
        fs      = {c: _make_feature(c, category="momentum") for c in cols}
        fs["tech_c"] = _make_feature("tech_c", category="trend")
        schema  = DatasetSchema(
            version="1.0.0", symbol=SYMBOL, features=fs,
            created_at="", frozen=False, schema_hash="",
        )
        catalog = FeatureCatalog.from_schema(schema)
        assert len(catalog.search(category="momentum")) == 2
        assert len(catalog.search(category="trend")) == 1

    def test_search_by_query(self):
        schema  = _make_schema(cols=["h1_rsi_14", "h1_macd", "m15_atr"])
        catalog = FeatureCatalog.from_schema(schema)
        results = catalog.search(query="rsi")
        assert len(results) == 1
        assert results[0].name == "h1_rsi_14"

    def test_to_json_roundtrip(self, tmp_path):
        schema  = _make_schema()
        catalog = FeatureCatalog.from_schema(schema)
        path    = tmp_path / "catalog.json"
        catalog.to_json(path)
        loaded = FeatureCatalog.from_json(path)
        assert loaded.feature_count == catalog.feature_count

    def test_to_markdown(self, tmp_path):
        schema  = _make_schema()
        catalog = FeatureCatalog.from_schema(schema)
        path    = tmp_path / "catalog.md"
        catalog.to_markdown(path)
        content = path.read_text()
        assert "Feature Catalog" in content

    def test_list_prefixes(self):
        schema  = _make_schema(cols=["h1_a", "m15_b"])
        catalog = FeatureCatalog.from_schema(schema)
        assert "h1_"  in catalog.list_prefixes()
        assert "m15_" in catalog.list_prefixes()


# =============================================================================
# 10. DatasetManifest / ManifestIndex
# =============================================================================


class TestDatasetManifest:
    def test_generate(self):
        df     = _make_df()
        schema = _make_schema()
        m      = DatasetManifest.generate(df, schema, dataset_version=1, file_path=Path("x.parquet"))
        assert m.dataset_version == 1
        assert m.feature_count   == schema.feature_count
        assert m.rows            == len(df)

    def test_save_load(self, tmp_path):
        df     = _make_df()
        schema = _make_schema()
        m      = DatasetManifest.generate(df, schema, 1, tmp_path / "x.parquet")
        path   = tmp_path / "manifest.json"
        m.save(path)
        loaded = DatasetManifest.load(path)
        assert loaded.dataset_version == 1

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(ManifestError):
            DatasetManifest.load(tmp_path / "nonexistent.json")

    def test_to_dict_roundtrip(self):
        df     = _make_df()
        schema = _make_schema()
        m      = DatasetManifest.generate(df, schema, 1, Path("x.parquet"))
        m2     = DatasetManifest.from_dict(m.to_dict())
        assert m2.dataset_version == 1


class TestManifestIndex:
    def test_empty_on_new(self, tmp_path):
        idx = ManifestIndex(tmp_path)
        assert idx.latest_version() is None
        assert idx.next_version() == 1

    def test_register_and_latest(self, tmp_path):
        df     = _make_df()
        schema = _make_schema()
        m      = DatasetManifest.generate(df, schema, 1, Path("x.parquet"))
        idx    = ManifestIndex(tmp_path)
        idx.register(m)
        assert idx.latest_version() == 1
        assert idx.next_version()   == 2

    def test_persist_across_instances(self, tmp_path):
        df     = _make_df()
        schema = _make_schema()
        for v in [1, 2, 3]:
            m   = DatasetManifest.generate(df, schema, v, Path(f"x_v{v}.parquet"))
            idx = ManifestIndex(tmp_path)
            idx.register(m)
        idx2 = ManifestIndex(tmp_path)
        assert idx2.versions() == [1, 2, 3]


# =============================================================================
# 11. FeatureLoader
# =============================================================================


class TestFeatureLoader:
    def test_save_creates_parquet(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        df     = _make_df()
        schema = _make_schema()
        path, version = loader.save(df, SYMBOL, schema)
        assert path.exists()
        assert version == 1

    def test_second_save_increments_version(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        df     = _make_df()
        schema = _make_schema()
        _, v1  = loader.save(df, SYMBOL, schema)
        _, v2  = loader.save(df, SYMBOL, schema)
        assert v1 == 1
        assert v2 == 2

    def test_load_version(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        df     = _make_df()
        schema = _make_schema()
        _, v   = loader.save(df, SYMBOL, schema)
        loaded = loader.load_version(SYMBOL, v)
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)

    def test_load_preserves_datetimeindex(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        df     = _make_df()
        schema = _make_schema()
        loader.save(df, SYMBOL, schema)
        loaded = loader.load_latest(SYMBOL)
        assert isinstance(loaded.index, pd.DatetimeIndex)

    def test_load_version_missing_raises(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        with pytest.raises(DatasetNotFoundError):
            loader.load_version(SYMBOL, 99)

    def test_load_latest_no_data_raises(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        with pytest.raises(DatasetNotFoundError):
            loader.load_latest(SYMBOL)

    def test_load_subset_column_pruning(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        df     = _make_df(cols=["h1_a", "m15_b", "tech_c"])
        schema = _make_schema(cols=["h1_a", "m15_b", "tech_c"])
        loader.save(df, SYMBOL, schema)
        loaded = loader.load_subset(SYMBOL, ["h1_a"])
        assert list(loaded.columns) == ["h1_a"]

    def test_list_versions(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        df     = _make_df()
        schema = _make_schema()
        loader.save(df, SYMBOL, schema)
        loader.save(df, SYMBOL, schema)
        assert loader.list_versions(SYMBOL) == [1, 2]

    def test_archive(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        df     = _make_df()
        schema = _make_schema()
        loader.save(df, SYMBOL, schema)
        dst = loader.archive(SYMBOL, 1)
        assert dst.exists()
        assert not (tmp_path / SYMBOL / "feature_dataset_v1.parquet").exists()

    def test_list_symbols(self, tmp_path):
        loader = FeatureLoader(tmp_path)
        df     = _make_df()
        schema = _make_schema()
        loader.save(df, "EURUSD", schema)
        loader.save(df, "GBPUSD", _make_schema(symbol="GBPUSD"))
        assert set(loader.list_symbols()) == {"EURUSD", "GBPUSD"}


# =============================================================================
# 12. FeatureStore — core API
# =============================================================================


class TestFeatureStoreCore:
    @pytest.fixture
    def store(self, tmp_path):
        return FeatureStore(
            base_dir=tmp_path / "features",
            schema_dir=tmp_path / "schemas",
            enable_hash_verification=False,
            enable_schema_validation=False,
        )

    def test_save_returns_manifest(self, store):
        df     = _make_df()
        schema = _make_schema()
        m      = store.save(df, SYMBOL, schema)
        assert isinstance(m, DatasetManifest)
        assert m.dataset_version == 1

    def test_save_auto_freezes_schema(self, store):
        df     = _make_df()
        schema = _make_schema()
        assert not schema.frozen
        store.save(df, SYMBOL, schema)
        loaded_schema = store.get_schema(SYMBOL)
        assert loaded_schema.frozen

    def test_load_latest(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        loaded = store.load_latest(SYMBOL)
        assert isinstance(loaded, pd.DataFrame)
        assert set(loaded.columns) == set(df.columns)

    def test_load_alias(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        assert store.load(SYMBOL) is not None

    def test_multiple_saves_increment_version(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        m2 = store.save(df, SYMBOL, schema)
        assert m2.dataset_version == 2

    def test_load_version(self, store):
        df1 = _make_df(5)
        df2 = _make_df(10)
        schema = _make_schema()
        store.save(df1, SYMBOL, schema)
        store.save(df2, SYMBOL, schema)
        assert len(store.load_version(SYMBOL, 1)) == 5
        assert len(store.load_version(SYMBOL, 2)) == 10

    def test_load_subset(self, store):
        df     = _make_df(cols=["h1_a", "m15_b", "tech_c"])
        schema = _make_schema(cols=["h1_a", "m15_b", "tech_c"])
        store.save(df, SYMBOL, schema)
        loaded = store.load_subset(SYMBOL, ["h1_a"])
        assert list(loaded.columns) == ["h1_a"]

    def test_load_category(self, store):
        cols   = ["h1_a", "m15_b"]
        fs     = {c: _make_feature(c, category="momentum") for c in cols}
        fs["tech_c"] = _make_feature("tech_c", category="trend")
        schema = DatasetSchema(
            version="1.0.0", symbol=SYMBOL, features=fs,
            created_at="", frozen=False,
            schema_hash=compute_features_hash(fs),
        )
        df = pd.DataFrame(
            {c: [1.0] for c in fs},
            index=pd.date_range("2024-01-01", periods=1, freq="15min", tz="UTC"),
        )
        store.save(df, SYMBOL, schema)
        loaded = store.load_category(SYMBOL, "momentum")
        assert set(loaded.columns) == {"h1_a", "m15_b"}

    def test_list_versions(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        store.save(df, SYMBOL, schema)
        assert store.list_versions(SYMBOL) == [1, 2]

    def test_list_features(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        feats = store.list_features(SYMBOL)
        assert set(feats) == set(df.columns)

    def test_list_symbols(self, store):
        df  = _make_df()
        s1  = _make_schema(symbol="EURUSD")
        s2  = _make_schema(symbol="GBPUSD")
        store.save(df, "EURUSD", s1)
        store.save(df, "GBPUSD", s2)
        assert set(store.list_symbols()) == {"EURUSD", "GBPUSD"}

    def test_get_manifest(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        m = store.get_manifest(SYMBOL, 1)
        assert m.dataset_version == 1

    def test_list_manifests(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        store.save(df, SYMBOL, schema)
        manifests = store.list_manifests(SYMBOL)
        assert [m.dataset_version for m in manifests] == [1, 2]

    def test_describe(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        info = store.describe(SYMBOL)
        assert info["symbol"]         == SYMBOL
        assert info["latest_dataset"] == 1
        assert info["feature_count"]  == schema.feature_count


# =============================================================================
# 13. FeatureStore — schema management
# =============================================================================


class TestFeatureStoreSchemas:
    @pytest.fixture
    def store(self, tmp_path):
        return FeatureStore(
            base_dir=tmp_path / "features",
            schema_dir=tmp_path / "schemas",
            enable_hash_verification=False,
            enable_schema_validation=False,
        )

    def test_register_and_get_schema(self, store):
        schema = _make_schema()
        store.register_schema(schema)
        loaded = store.get_schema(SYMBOL)
        assert loaded.version == "1.0.0"

    def test_freeze_schema(self, store):
        schema = _make_schema()
        store.register_schema(schema)
        frozen = store.freeze_schema(SYMBOL)
        assert frozen.frozen

    def test_compare_schemas(self, store):
        s1 = _make_schema(cols=["h1_a", "h1_b"])
        s2 = _make_schema(cols=["h1_a", "h1_c"], version="1.1.0")
        store.register_schema(s1)
        store.register_schema(s2)
        diff = store.compare_schemas(SYMBOL, "1.0.0", "1.1.0")
        assert "h1_c" in diff["added"]

    def test_rollback_schema(self, store):
        store.register_schema(_make_schema(version="1.0.0"))
        store.register_schema(_make_schema(version="1.1.0"))
        store.rollback_schema(SYMBOL, "1.0.0")
        assert store.get_schema(SYMBOL).version == "1.0.0"

    def test_list_schema_versions(self, store):
        store.register_schema(_make_schema(version="1.0.0"))
        store.register_schema(_make_schema(version="1.1.0"))
        assert store.list_schema_versions(SYMBOL) == ["1.0.0", "1.1.0"]

    def test_check_compatibility(self, store):
        s1 = _make_schema(cols=["h1_a"])
        s2 = _make_schema(cols=["h1_a", "h1_b"], version="1.1.0")
        store.register_schema(s1)
        store.register_schema(s2)
        report = store.check_compatibility(SYMBOL, "1.0.0", "1.1.0")
        assert report.is_compatible

    def test_get_migration_plan(self, store):
        s1 = _make_schema(cols=["h1_a", "h1_b"])
        s2 = _make_schema(cols=["h1_a", "h1_c"], version="2.0.0")
        store.register_schema(s1)
        store.register_schema(s2)
        plan = store.get_migration_plan(SYMBOL, "1.0.0", "2.0.0")
        assert "actions" in plan


# =============================================================================
# 14. FeatureStore — validation
# =============================================================================


class TestFeatureStoreValidation:
    @pytest.fixture
    def store_strict(self, tmp_path):
        return FeatureStore(
            base_dir=tmp_path / "features",
            schema_dir=tmp_path / "schemas",
            enable_hash_verification=False,
            enable_schema_validation=True,
        )

    def test_validate_passes(self, store_strict):
        df     = _make_df()
        schema = _make_schema()
        store_strict.save(df, SYMBOL, schema)
        report = store_strict.validate(df, SYMBOL)
        assert report.is_valid

    def test_validate_missing_column(self, store_strict):
        df_full   = _make_df(cols=["h1_a", "m15_b"])
        schema    = _make_schema(cols=["h1_a", "m15_b"])
        store_strict.save(df_full, SYMBOL, schema)
        df_partial = _make_df(cols=["h1_a"])
        report    = store_strict.validate(df_partial, SYMBOL)
        assert not report.is_valid

    def test_validate_or_raise_raises(self, store_strict):
        df_full  = _make_df(cols=["h1_a", "m15_b"])
        schema   = _make_schema(cols=["h1_a", "m15_b"])
        store_strict.save(df_full, SYMBOL, schema)
        df_bad   = _make_df(cols=["h1_a"])
        with pytest.raises(SchemaMismatchError):
            store_strict.validate_or_raise(df_bad, SYMBOL)

    def test_hash_verification_mismatch_raises(self, tmp_path):
        store1 = FeatureStore(
            tmp_path / "f1", tmp_path / "s1",
            enable_hash_verification=True,
            enable_schema_validation=False,
        )
        df     = _make_df()
        schema = _make_schema()
        store1.save(df, SYMBOL, schema)
        # Corrupt the saved hash file
        hash_file = tmp_path / "f1" / SYMBOL / "schema.hash"
        hash_file.write_text("0" * 64, encoding="utf-8")
        with pytest.raises(HashVerificationError):
            store1.validate(df, SYMBOL)

    def test_verify_schema_hash_matching(self):
        store = FeatureStore.__new__(FeatureStore)
        s1 = _make_schema()
        s2 = _make_schema()  # identical features → same hash
        # Both have the same features so hashes match
        assert store.verify_schema_hash(s1, s2)

    def test_verify_schema_hash_mismatch(self):
        store = FeatureStore.__new__(FeatureStore)
        s1 = _make_schema(cols=["h1_a"])
        s2 = _make_schema(cols=["h1_b"])
        with pytest.raises(HashVerificationError):
            store.verify_schema_hash(s1, s2)


# =============================================================================
# 15. FeatureStore — catalog integration
# =============================================================================


class TestFeatureStoreCatalog:
    @pytest.fixture
    def store(self, tmp_path):
        return FeatureStore(
            base_dir=tmp_path / "features",
            schema_dir=tmp_path / "schemas",
            enable_hash_verification=False,
            enable_schema_validation=False,
        )

    def test_build_catalog(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        catalog = store.build_catalog(SYMBOL)
        assert catalog.feature_count == schema.feature_count

    def test_search_features(self, store):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        results = store.search_features(SYMBOL, prefix="h1_")
        assert all(f.prefix == "h1_" for f in results)

    def test_save_catalog_json(self, store, tmp_path):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        path   = tmp_path / "catalog.json"
        store.save_catalog(SYMBOL, json_path=path)
        assert path.exists()

    def test_save_catalog_markdown(self, store, tmp_path):
        df     = _make_df()
        schema = _make_schema()
        store.save(df, SYMBOL, schema)
        path   = tmp_path / "catalog.md"
        store.save_catalog(SYMBOL, markdown_path=path)
        assert path.exists()
        assert "Feature Catalog" in path.read_text()


# =============================================================================
# 16. Schema / Feature Contracts
# =============================================================================


class TestSchemaContract:
    def test_satisfied_contract(self):
        schema = _make_schema(cols=["h1_a", "m15_b"])
        contract = SchemaContract(
            required_features=["h1_a"],
            required_prefixes=["m15_"],
            min_feature_count=2,
        )
        report = contract.validate(schema)
        assert report.is_valid

    def test_missing_feature_violation(self):
        schema   = _make_schema(cols=["h1_a"])
        contract = SchemaContract(required_features=["m15_missing"])
        report   = contract.validate(schema)
        assert not report.is_valid

    def test_validate_or_raise(self):
        schema   = _make_schema(cols=["h1_a"])
        contract = SchemaContract(required_features=["m15_x"], name="test")
        with pytest.raises(ValidationError):
            contract.validate_or_raise(schema)


class TestFeatureContract:
    def test_validate_output_passes(self):
        output_schema = {"h1_rsi_14": _make_feature("h1_rsi_14")}
        contract = FeatureContract(
            module_name="MockEngine",
            output_schemas=output_schema,
        )
        df = pd.DataFrame({"h1_rsi_14": [0.5, 0.6]})
        report = contract.validate_output(df, strict=False)
        # Not strict — missing columns in df outside output_schemas don't fail
        assert isinstance(report, ValidationReport)

    def test_validate_input_missing(self):
        contract = FeatureContract(
            module_name="MockEngine",
            input_features=["close", "volume"],
        )
        df      = pd.DataFrame({"close": [1.0]})
        missing = contract.validate_input(df)
        assert "volume" in missing


# =============================================================================
# 17. Integration — full round-trip
# =============================================================================


class TestIntegrationRoundTrip:
    def test_full_pipeline(self, tmp_path):
        store = FeatureStore(
            base_dir=tmp_path / "features",
            schema_dir=tmp_path / "schemas",
            enable_hash_verification=False,
            enable_schema_validation=True,
        )
        df     = _make_df(100)
        schema = _make_schema()

        # Save
        m = store.save(df, SYMBOL, schema)
        assert m.dataset_version == 1

        # Load latest
        loaded = store.load_latest(SYMBOL)
        assert len(loaded) == 100
        pd.testing.assert_index_equal(loaded.index, df.index)

        # Validate
        report = store.validate(loaded, SYMBOL)
        assert report.is_valid

        # Catalog
        catalog = store.build_catalog(SYMBOL)
        assert catalog.feature_count == schema.feature_count

        # Schema round-trip
        s = store.get_schema(SYMBOL)
        assert s.frozen
        assert s.schema_hash == schema.schema_hash

        # Describe
        info = store.describe(SYMBOL)
        assert info["latest_dataset"] == 1

    def test_schema_evolution_round_trip(self, tmp_path):
        store = FeatureStore(
            base_dir=tmp_path / "features",
            schema_dir=tmp_path / "schemas",
            enable_hash_verification=False,
            enable_schema_validation=False,
        )
        df1    = _make_df(10, cols=["h1_a", "m15_b"])
        s1     = _make_schema(cols=["h1_a", "m15_b"])
        store.save(df1, SYMBOL, s1)

        # Evolve schema
        df2    = _make_df(10, cols=["h1_a", "m15_b", "h1_c"])
        s2     = _make_schema(cols=["h1_a", "m15_b", "h1_c"], version="1.1.0")
        store.save(df2, SYMBOL, s2)

        versions = store.list_versions(SYMBOL)
        assert versions == [1, 2]

        schema_versions = store.list_schema_versions(SYMBOL)
        assert "1.1.0" in schema_versions

        v1_df = store.load_version(SYMBOL, 1)
        assert "h1_c" not in v1_df.columns


# =============================================================================
# 18. Performance
# =============================================================================


class TestPerformance:
    def test_save_load_100k_rows_under_5s(self, tmp_path):
        store  = FeatureStore(
            base_dir=tmp_path / "features",
            schema_dir=tmp_path / "schemas",
            enable_hash_verification=False,
            enable_schema_validation=False,
        )
        rng    = np.random.default_rng(0)
        n      = 100_000
        cols   = [f"h1_feat_{i:03d}" for i in range(50)]
        idx    = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
        df     = pd.DataFrame(rng.random((n, len(cols))), index=idx, columns=cols)
        schema = _make_schema(cols=cols)

        t0 = time.perf_counter()
        store.save(df, SYMBOL, schema)
        loaded = store.load_latest(SYMBOL)
        elapsed = time.perf_counter() - t0

        assert len(loaded) == n
        assert elapsed < 5.0, f"save+load took {elapsed:.2f}s — expected < 5s"
