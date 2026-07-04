"""Feature Store exception hierarchy."""

from __future__ import annotations


class FeatureStoreError(Exception):
    """Base exception for all Feature Store errors."""


class SchemaMismatchError(FeatureStoreError):
    """Schema hash at inference time does not match training-time hash."""


class FeatureNotFoundError(FeatureStoreError):
    """Requested feature does not exist in the catalog or dataset."""


class InvalidPrefixError(FeatureStoreError):
    """Feature name uses an unrecognised or forbidden prefix."""


class InvalidDatatypeError(FeatureStoreError):
    """Feature dtype conflicts with the schema contract."""


class VersionMismatchError(FeatureStoreError):
    """Schema or dataset version incompatibility detected."""


class CompatibilityError(FeatureStoreError):
    """Breaking change between schema versions prevents safe loading."""


class DependencyError(FeatureStoreError):
    """A required upstream feature or module is missing."""


class SchemaFrozenError(FeatureStoreError):
    """Attempted to mutate a frozen (published) schema."""


class SchemaNotFoundError(FeatureStoreError):
    """No schema found for the requested symbol / version."""


class DatasetNotFoundError(FeatureStoreError):
    """No dataset file found for the requested symbol / version."""


class ValidationError(FeatureStoreError):
    """DataFrame failed schema validation."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors: list[str] = errors or []


class ManifestError(FeatureStoreError):
    """Manifest file is missing, corrupt, or inconsistent."""


class CatalogError(FeatureStoreError):
    """Feature catalog operation failed."""


class HashVerificationError(FeatureStoreError):
    """SHA-256 schema hash verification failed."""
