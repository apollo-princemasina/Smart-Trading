"""
FeatureStore — central data layer for the Smart Trading feature pipeline.

Every downstream component (training, backtesting, inference) must consume
data through this API.  Direct Parquet reads are prohibited.

Design principles
-----------------
* **Immutable datasets** — each ``save()`` call writes a new versioned file;
  old files are never overwritten.
* **Schema-first** — every dataset is paired with a :class:`DatasetSchema`.
  Schemas are frozen after first publication; evolution creates new versions.
* **SHA-256 hash verification** — training/inference schemas are compared by
  their structural hash so column-set drift is detected immediately.
* **Manifest tracking** — a ``manifest_index.json`` and per-version manifest
  JSON file accompany every saved dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .compatibility import CompatibilityChecker, CompatibilityReport
from .dataset_hash import compute_features_hash
from .dataset_manifest import DatasetManifest
from .exceptions import (
    DatasetNotFoundError,
    HashVerificationError,
    SchemaNotFoundError,
    SchemaMismatchError,
)
from .feature_catalog import FeatureCatalog
from .feature_loader import FeatureLoader
from .schema import DatasetSchema, FeatureSchema
from .schema_manager import SchemaManager
from .schema_registry import SchemaRegistry
from .schema_validator import SchemaValidator, ValidationReport

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Central data-access layer for the Smart Trading feature pipeline.

    Parameters
    ----------
    base_dir:
        Root directory for all feature store data
        (e.g. ``data/features``).
    schema_dir:
        Root directory for schema JSON files
        (e.g. ``data/schemas``).
    pipeline_version:
        Current pipeline version injected into every schema / manifest.
    author:
        Default author tag for auto-inferred schemas.
    enable_hash_verification:
        When True, ``validate()`` compares the schema hash of the loaded
        schema against the hash stored at save time and raises
        :class:`HashVerificationError` on mismatch.
    enable_schema_validation:
        When True, ``validate()`` and ``validate_or_raise()`` run the full
        :class:`SchemaValidator` check.

    Usage
    -----
    ::

        store = FeatureStore(base_dir="data/features", schema_dir="data/schemas")

        # Save
        manifest = store.save(fused_df, symbol="EURUSD", schema=my_schema)

        # Load
        df = store.load_latest("EURUSD")

        # Schema management
        store.register_schema(schema)
        schema = store.get_schema("EURUSD")
        store.freeze_schema("EURUSD")

        # Catalog
        catalog = store.build_catalog("EURUSD")
        features = store.search_features("EURUSD", category="momentum")
    """

    def __init__(
        self,
        base_dir:                str | Path,
        schema_dir:              str | Path,
        pipeline_version:        str  = "1.0.0",
        author:                  str  = "SmartTrading",
        enable_hash_verification: bool = True,
        enable_schema_validation: bool = True,
    ):
        self._base_dir    = Path(base_dir)
        self._schema_dir  = Path(schema_dir)
        self._pipeline_ver = pipeline_version
        self._author      = author
        self._verify_hash  = enable_hash_verification
        self._validate     = enable_schema_validation

        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._schema_dir.mkdir(parents=True, exist_ok=True)

        self._loader    = FeatureLoader(self._base_dir)
        self._registry  = SchemaRegistry(self._schema_dir)
        self._manager   = SchemaManager(
            self._registry, pipeline_version, author
        )
        self._validator  = SchemaValidator()
        self._compat     = CompatibilityChecker()

    # =========================================================================
    # Dataset I/O
    # =========================================================================

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        schema: DatasetSchema,
        auto_version: bool = True,
    ) -> DatasetManifest:
        """
        Save *df* as the next immutable Parquet version for *symbol*.

        Parameters
        ----------
        df:
            Feature DataFrame with a :class:`~pandas.DatetimeIndex`.
        symbol:
            Instrument identifier (e.g. ``"EURUSD"``).
        schema:
            :class:`DatasetSchema` describing the columns.  If *schema* is not
            yet frozen and no existing frozen schema exists, it is frozen
            automatically.
        auto_version:
            Ignored (always increments the dataset version).  Present for
            API consistency.

        Returns
        -------
        :class:`DatasetManifest`
            Metadata record for the saved dataset.
        """
        # Auto-freeze schema if not already frozen
        if not schema.frozen:
            schema = self._manager.freeze(schema)
        else:
            # Ensure it exists in the registry
            if not self._registry.exists(symbol, schema.version):
                self._registry.register(schema)

        path, version = self._loader.save(df, symbol, schema)
        manifest      = self._loader.get_manifest(symbol, version)

        logger.info(
            "FeatureStore.save: %s v%d (%d rows × %d cols) → %s",
            symbol, version, len(df), df.shape[1], path,
        )
        return manifest

    def load(self, symbol: str) -> pd.DataFrame:
        """Alias for :meth:`load_latest`."""
        return self.load_latest(symbol)

    def load_latest(self, symbol: str) -> pd.DataFrame:
        """Return the most recently saved dataset for *symbol*."""
        return self._loader.load_latest(symbol)

    def load_version(self, symbol: str, version: int) -> pd.DataFrame:
        """Return dataset *version* for *symbol*."""
        return self._loader.load_version(symbol, version)

    def load_schema_version(
        self,
        symbol: str,
        schema_version: str,
        dataset_version: int | None = None,
    ) -> pd.DataFrame:
        """
        Load the dataset (optionally a specific *dataset_version*) that was
        saved under *schema_version*.

        If *dataset_version* is None, returns the latest dataset whose manifest
        references *schema_version*.
        """
        manifests = self._loader.list_manifests(symbol)
        candidates = [m for m in manifests if m.schema_version == schema_version]
        if not candidates:
            raise DatasetNotFoundError(
                f"No dataset found for {symbol!r} with schema version {schema_version!r}"
            )
        if dataset_version is not None:
            for m in candidates:
                if m.dataset_version == dataset_version:
                    return self._loader.load_version(symbol, dataset_version)
            raise DatasetNotFoundError(
                f"Dataset v{dataset_version} for {symbol!r} does not use schema "
                f"version {schema_version!r}"
            )
        # Return the highest version
        best = max(candidates, key=lambda m: m.dataset_version)
        return self._loader.load_version(symbol, best.dataset_version)

    def load_subset(
        self,
        symbol: str,
        features: list[str],
        version: int | None = None,
    ) -> pd.DataFrame:
        """Load only *features* (columns) from the dataset (Parquet pruning)."""
        return self._loader.load_subset(symbol, features, version)

    def load_category(
        self,
        symbol: str,
        category: str,
        version: int | None = None,
    ) -> pd.DataFrame:
        """Load all features belonging to *category* from the dataset."""
        schema  = self._registry.get_latest(symbol)
        columns = [
            name for name, fs in schema.features.items()
            if fs.category == category
        ]
        if not columns:
            return pd.DataFrame()
        return self._loader.load_subset(symbol, columns, version)

    def load_timeframe(
        self,
        symbol: str,
        timeframe: str,
        version: int | None = None,
    ) -> pd.DataFrame:
        """Load all features for *timeframe* from the dataset."""
        schema  = self._registry.get_latest(symbol)
        columns = [
            name for name, fs in schema.features.items()
            if fs.timeframe == timeframe
        ]
        if not columns:
            return pd.DataFrame()
        return self._loader.load_subset(symbol, columns, version)

    # =========================================================================
    # Schema management
    # =========================================================================

    def register_schema(self, schema: DatasetSchema) -> None:
        """Register *schema* in the schema registry."""
        self._registry.register(schema)
        logger.info("Registered schema %s v%s", schema.symbol, schema.version)

    def get_schema(self, symbol: str, version: str = "latest") -> DatasetSchema:
        """Return the schema for *symbol* at *version* (or ``"latest"``)."""
        return self._manager.load(symbol, version)

    def freeze_schema(self, symbol: str, version: str = "latest") -> DatasetSchema:
        """Freeze the schema for *symbol* / *version*."""
        schema = self._manager.load(symbol, version)
        return self._manager.freeze(schema)

    def compare_schemas(
        self,
        symbol: str,
        version_a: str,
        version_b: str,
    ) -> dict:
        """Return a structured diff between two schema versions."""
        a = self._registry.get(symbol, version_a)
        b = self._registry.get(symbol, version_b)
        return self._manager.compare(a, b)

    def rollback_schema(self, symbol: str, to_version: str) -> DatasetSchema:
        """Pin *to_version* as the active schema for *symbol*."""
        return self._manager.rollback(symbol, to_version)

    def infer_and_register_schema(
        self,
        df: pd.DataFrame,
        symbol: str,
        version: str = "1.0.0",
        freeze: bool = True,
        **kwargs: Any,
    ) -> DatasetSchema:
        """
        Auto-infer a schema from *df*, optionally freeze it, and register it.
        """
        schema = self._manager.infer_from_dataframe(df, symbol, version, **kwargs)
        if freeze:
            schema = self._manager.freeze(schema)
        else:
            self._registry.register(schema)
        return schema

    def check_compatibility(
        self,
        symbol: str,
        version_a: str,
        version_b: str,
    ) -> CompatibilityReport:
        """Return a :class:`CompatibilityReport` for two schema versions."""
        a = self._registry.get(symbol, version_a)
        b = self._registry.get(symbol, version_b)
        return self._compat.check(a, b)

    def get_migration_plan(
        self,
        symbol: str,
        from_version: str,
        to_version: str,
    ) -> dict:
        """Return a migration plan dict to move from one schema to another."""
        a = self._registry.get(symbol, from_version)
        b = self._registry.get(symbol, to_version)
        return self._compat.get_migration_plan(a, b)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, df: pd.DataFrame, symbol: str) -> ValidationReport:
        """
        Validate *df* against the latest frozen schema for *symbol*.

        If hash verification is enabled, raises :class:`HashVerificationError`
        when the computed hash of *df*'s columns does not match the saved schema
        hash.
        """
        schema = self._registry.get_latest(symbol)

        if self._verify_hash:
            saved_hash    = self._loader.load_schema_hash(symbol)
            computed_hash = schema.schema_hash
            if saved_hash and saved_hash != computed_hash:
                raise HashVerificationError(
                    f"Schema hash mismatch for {symbol!r}: "
                    f"saved={saved_hash[:12]}… computed={computed_hash[:12]}…"
                )

        if self._validate:
            return self._validator.validate(df, schema)

        return ValidationReport(is_valid=True)

    def validate_or_raise(self, df: pd.DataFrame, symbol: str) -> None:
        """Validate and raise :class:`SchemaMismatchError` if the check fails."""
        report = self.validate(df, symbol)
        if not report.is_valid:
            raise SchemaMismatchError(
                f"DataFrame for {symbol!r} does not conform to the registered schema: "
                + "; ".join(report.errors[:5])
            )

    def verify_schema_hash(
        self,
        training_schema: DatasetSchema,
        inference_schema: DatasetSchema,
    ) -> bool:
        """
        Compare the structural hashes of two schemas.

        Raises :class:`HashVerificationError` if they differ.
        Returns True if they match.
        """
        if training_schema.schema_hash != inference_schema.schema_hash:
            raise HashVerificationError(
                f"Schema hash mismatch between training ({training_schema.version}) "
                f"and inference ({inference_schema.version}) schemas for "
                f"{training_schema.symbol!r}. "
                "Feature set differs — retrain or align schemas before inference."
            )
        return True

    # =========================================================================
    # Feature Catalog
    # =========================================================================

    def build_catalog(self, symbol: str) -> FeatureCatalog:
        """Build and return a searchable :class:`FeatureCatalog` for *symbol*."""
        schema  = self._registry.get_latest(symbol)
        catalog = FeatureCatalog.from_schema(schema)
        return catalog

    def search_features(
        self,
        symbol: str,
        category:     str | None = None,
        prefix:       str | None = None,
        timeframe:    str | None = None,
        source_module: str | None = None,
        tag:          str | None = None,
        deprecated:   bool | None = None,
        query:        str | None = None,
    ) -> list[FeatureSchema]:
        """Search features in the latest schema for *symbol*."""
        catalog = self.build_catalog(symbol)
        return catalog.search(
            category=category,
            prefix=prefix,
            timeframe=timeframe,
            source_module=source_module,
            tag=tag,
            deprecated=deprecated,
            query=query,
        )

    def save_catalog(
        self,
        symbol: str,
        json_path: Path | None = None,
        markdown_path: Path | None = None,
    ) -> FeatureCatalog:
        """Build the catalog and optionally write it to disk."""
        catalog = self.build_catalog(symbol)
        if json_path:
            catalog.to_json(json_path)
        if markdown_path:
            catalog.to_markdown(markdown_path)
        return catalog

    # =========================================================================
    # Listing / introspection
    # =========================================================================

    def list_versions(self, symbol: str) -> list[int]:
        """Return all saved dataset version numbers for *symbol* (ascending)."""
        return self._loader.list_versions(symbol)

    def list_features(self, symbol: str, version: str = "latest") -> list[str]:
        """Return all feature names in the schema for *symbol*."""
        schema = self._manager.load(symbol, version)
        return schema.feature_names

    def list_symbols(self) -> list[str]:
        """Return all symbols that have at least one saved dataset."""
        return self._loader.list_symbols()

    def list_schema_versions(self, symbol: str) -> list[str]:
        """Return all registered schema versions for *symbol* (ascending)."""
        return self._registry.list_versions(symbol)

    # =========================================================================
    # Manifest access
    # =========================================================================

    def get_manifest(self, symbol: str, version: int) -> DatasetManifest:
        """Return the manifest for dataset *version* of *symbol*."""
        return self._loader.get_manifest(symbol, version)

    def list_manifests(self, symbol: str) -> list[DatasetManifest]:
        """Return all manifests for *symbol*, ordered by dataset version."""
        return self._loader.list_manifests(symbol)

    # =========================================================================
    # Convenience / introspection
    # =========================================================================

    def describe(self, symbol: str) -> dict:
        """
        Return a summary dict for *symbol*:
        dataset versions, latest schema version, feature count, etc.
        """
        dataset_versions = self.list_versions(symbol)

        try:
            schema = self._registry.get_latest(symbol)
        except SchemaNotFoundError:
            schema = None

        try:
            manifests = self.list_manifests(symbol)
            latest_manifest = manifests[-1] if manifests else None
        except Exception:
            latest_manifest = None

        return {
            "symbol":           symbol,
            "dataset_versions": dataset_versions,
            "latest_dataset":   dataset_versions[-1] if dataset_versions else None,
            "schema_version":   schema.version if schema else None,
            "feature_count":    schema.feature_count if schema else None,
            "schema_frozen":    schema.frozen if schema else None,
            "schema_hash":      schema.schema_hash if schema else None,
            "latest_manifest":  latest_manifest.to_dict() if latest_manifest else None,
        }

    def __repr__(self) -> str:
        symbols = self.list_symbols()
        return (
            f"FeatureStore(base={self._base_dir}, "
            f"symbols={symbols}, pipeline={self._pipeline_ver})"
        )
