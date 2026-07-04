"""SHA-256 schema hashing for training/inference reproducibility.

The structural hash covers only those fields that affect model consumption:
feature names, their ordering, dtypes, prefixes, and nullability.
Purely descriptive metadata (description, author, tags) is excluded so
that documentation updates do not invalidate trained models.
"""

from __future__ import annotations

import hashlib
import json

from .schema import DatasetSchema, FeatureSchema


# ── Internal helpers ──────────────────────────────────────────────────────────


def _structural_repr(features: dict[str, FeatureSchema]) -> str:
    """
    Build a canonical, deterministic string from structural schema fields.

    The string is the JSON serialisation of a sorted list of
    ``[name, prefix, dtype, nullable, min_value, max_value]`` tuples,
    preserving column *order* (insertion order of *features*).
    """
    entries = [
        {
            "name":      f.name,
            "prefix":    f.prefix,
            "dtype":     f.dtype,
            "nullable":  f.nullable,
            "min_value": f.min_value,
            "max_value": f.max_value,
        }
        for f in features.values()   # preserve column order
    ]
    return json.dumps(entries, sort_keys=True, separators=(",", ":"))


# ── Public API ────────────────────────────────────────────────────────────────


def compute_features_hash(features: dict[str, FeatureSchema]) -> str:
    """Return a 64-character SHA-256 hex digest of the structural schema."""
    payload = _structural_repr(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_schema_hash(schema: DatasetSchema) -> str:
    """Convenience wrapper: compute hash from a full ``DatasetSchema``."""
    return compute_features_hash(schema.features)


def verify_schema_hash(schema: DatasetSchema) -> bool:
    """
    Recompute the structural hash and compare to the stored value.

    Returns ``True`` if they match.  A ``False`` result means the schema
    struct was mutated after hashing (or the stored hash is stale).
    """
    return compute_schema_hash(schema) == schema.schema_hash


def hash_dataframe_columns(column_names: list[str], dtypes: dict[str, str]) -> str:
    """
    Hash the column list + dtypes of a concrete DataFrame.

    Useful for a quick check that an in-memory DataFrame matches the schema's
    expected column set before feeding it to a model.
    """
    payload = json.dumps(
        [{"name": n, "dtype": dtypes.get(n, "unknown")} for n in column_names],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_hash(hash_value: str, path) -> None:
    """Write *hash_value* to *path* (one line, no newline trimming needed)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hash_value, encoding="utf-8")


def load_hash(path) -> str:
    """Read and return the hash stored in *path*."""
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8").strip()
