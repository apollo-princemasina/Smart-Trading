"""
src.feature_store
=================

Central data layer for the Smart Trading feature pipeline.

Public API
----------
.. code-block:: python

    from src.feature_store import (
        FeatureStore,
        DatasetSchema, FeatureSchema,
        SchemaManager, SchemaRegistry,
        SchemaValidator, ValidationReport,
        CompatibilityChecker, CompatibilityReport,
        FeatureCatalog, FeatureLoader,
        DatasetManifest, ManifestIndex,
        # Exceptions
        FeatureStoreError, SchemaMismatchError, SchemaFrozenError,
        SchemaNotFoundError, DatasetNotFoundError,
        HashVerificationError, ValidationError,
        # Utilities
        compute_schema_hash, verify_schema_hash,
        validate_prefix, extract_prefix, VALID_PREFIXES,
        SemanticVersion, determine_version_bump,
    )
"""

from __future__ import annotations

# ── Core store ────────────────────────────────────────────────────────────────
from .feature_store import FeatureStore

# ── Schema model ─────────────────────────────────────────────────────────────
from .schema import DatasetSchema, FeatureSchema

# ── Schema lifecycle ──────────────────────────────────────────────────────────
from .schema_manager import SchemaManager
from .schema_registry import SchemaRegistry
from .schema_validator import SchemaValidator, ValidationReport

# ── Versioning ────────────────────────────────────────────────────────────────
from .schema_versioning import SemanticVersion, determine_version_bump

# ── Compatibility ─────────────────────────────────────────────────────────────
from .compatibility import CompatibilityChecker, CompatibilityReport

# ── Catalog ───────────────────────────────────────────────────────────────────
from .feature_catalog import FeatureCatalog

# ── Loader & manifest ────────────────────────────────────────────────────────
from .feature_loader import FeatureLoader
from .dataset_manifest import DatasetManifest, ManifestIndex

# ── Hashing ───────────────────────────────────────────────────────────────────
from .dataset_hash import compute_schema_hash, verify_schema_hash, compute_features_hash

# ── Prefix utilities ─────────────────────────────────────────────────────────
from .feature_prefix import (
    VALID_PREFIXES,
    extract_prefix,
    validate_prefix,
    validate_all_prefixes,
    group_by_prefix,
    is_valid_prefix,
)

# ── Contracts ─────────────────────────────────────────────────────────────────
from .feature_contract import FeatureContract
from .schema_contract import SchemaContract

# ── Exceptions ───────────────────────────────────────────────────────────────
from .exceptions import (
    FeatureStoreError,
    SchemaMismatchError,
    FeatureNotFoundError,
    InvalidPrefixError,
    InvalidDatatypeError,
    VersionMismatchError,
    CompatibilityError,
    DependencyError,
    SchemaFrozenError,
    SchemaNotFoundError,
    DatasetNotFoundError,
    ValidationError,
    ManifestError,
    CatalogError,
    HashVerificationError,
)

__all__ = [
    # Core
    "FeatureStore",
    # Schema model
    "DatasetSchema",
    "FeatureSchema",
    # Lifecycle
    "SchemaManager",
    "SchemaRegistry",
    "SchemaValidator",
    "ValidationReport",
    # Versioning
    "SemanticVersion",
    "determine_version_bump",
    # Compatibility
    "CompatibilityChecker",
    "CompatibilityReport",
    # Catalog
    "FeatureCatalog",
    # I/O
    "FeatureLoader",
    "DatasetManifest",
    "ManifestIndex",
    # Hashing
    "compute_schema_hash",
    "verify_schema_hash",
    "compute_features_hash",
    # Prefix
    "VALID_PREFIXES",
    "extract_prefix",
    "validate_prefix",
    "validate_all_prefixes",
    "group_by_prefix",
    "is_valid_prefix",
    # Contracts
    "FeatureContract",
    "SchemaContract",
    # Exceptions
    "FeatureStoreError",
    "SchemaMismatchError",
    "FeatureNotFoundError",
    "InvalidPrefixError",
    "InvalidDatatypeError",
    "VersionMismatchError",
    "CompatibilityError",
    "DependencyError",
    "SchemaFrozenError",
    "SchemaNotFoundError",
    "DatasetNotFoundError",
    "ValidationError",
    "ManifestError",
    "CatalogError",
    "HashVerificationError",
]
