"""Core schema data structures for the Feature Store.

``FeatureSchema``   — per-column metadata contract.
``DatasetSchema``   — collection of features + version + hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Per-feature schema ────────────────────────────────────────────────────────


@dataclass
class FeatureSchema:
    """Rich metadata contract for a single feature column.

    Every field that affects how a model consumes the feature (name, dtype,
    prefix, nullable, range) participates in the structural hash.  Purely
    descriptive fields (description, author, tags) do not affect the hash.
    """

    # ── Identity ─────────────────────────────────────────────
    name:           str                      # full name including prefix
    prefix:         str                      # e.g. "h1_"
    category:       str                      # e.g. "momentum"
    description:    str = ""
    source_module:  str = ""                 # engine class name
    timeframe:      str = ""                 # "H1", "D", "M15", …

    # ── Structure ────────────────────────────────────────────
    dependencies:   list[str]  = field(default_factory=list)
    dtype:          str        = "float64"   # numpy/pandas dtype string
    nullable:       bool       = True
    default_value:  Any        = None
    min_value:      float|None = None
    max_value:      float|None = None
    units:          str        = ""
    example_value:  Any        = None

    # ── Provenance ───────────────────────────────────────────
    version:        str        = "1.0.0"
    author:         str        = ""
    created_date:   str        = ""          # ISO-8601 date
    updated_date:   str        = ""
    deprecated:     bool       = False
    tags:           list[str]  = field(default_factory=list)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "prefix":        self.prefix,
            "category":      self.category,
            "description":   self.description,
            "source_module": self.source_module,
            "timeframe":     self.timeframe,
            "dependencies":  self.dependencies,
            "dtype":         self.dtype,
            "nullable":      self.nullable,
            "default_value": self.default_value,
            "min_value":     self.min_value,
            "max_value":     self.max_value,
            "units":         self.units,
            "example_value": self.example_value,
            "version":       self.version,
            "author":        self.author,
            "created_date":  self.created_date,
            "updated_date":  self.updated_date,
            "deprecated":    self.deprecated,
            "tags":          self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSchema:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def structural_tuple(self) -> tuple:
        """Tuple used for hash computation — structural fields only."""
        return (self.name, self.prefix, self.dtype, self.nullable, self.min_value, self.max_value)


# ── Dataset schema ────────────────────────────────────────────────────────────


@dataclass
class DatasetSchema:
    """Schema for a complete fused feature dataset.

    The ``schema_hash`` field is computed from the structural content of all
    feature schemas.  It changes whenever any structural field changes,
    serving as the ground truth for training/inference compatibility.
    """

    version:          str                        # semantic version "1.0.0"
    symbol:           str                        # e.g. "EURUSD"
    features:         dict[str, FeatureSchema]   # ordered: name → schema
    created_at:       str                        # ISO-8601 timestamp
    frozen:           bool          = False
    schema_hash:      str           = ""         # SHA-256, set after creation
    pipeline_version: str           = "1.0.0"
    description:      str           = ""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def prefixes(self) -> set[str]:
        return {f.prefix for f in self.features.values()}

    @property
    def categories(self) -> set[str]:
        return {f.category for f in self.features.values() if f.category}

    @property
    def timeframes(self) -> set[str]:
        return {f.timeframe for f in self.features.values() if f.timeframe}

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names (insertion order preserved)."""
        return list(self.features)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version":          self.version,
            "symbol":           self.symbol,
            "features":         {k: v.to_dict() for k, v in self.features.items()},
            "created_at":       self.created_at,
            "frozen":           self.frozen,
            "schema_hash":      self.schema_hash,
            "pipeline_version": self.pipeline_version,
            "description":      self.description,
            "feature_count":    self.feature_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DatasetSchema:
        features = {
            k: FeatureSchema.from_dict(v) for k, v in d.get("features", {}).items()
        }
        return cls(
            version          = d["version"],
            symbol           = d["symbol"],
            features         = features,
            created_at       = d.get("created_at", ""),
            frozen           = d.get("frozen", False),
            schema_hash      = d.get("schema_hash", ""),
            pipeline_version = d.get("pipeline_version", "1.0.0"),
            description      = d.get("description", ""),
        )

    def to_json(self, path: Path, indent: int = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> DatasetSchema:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_markdown(self, path: Path) -> None:
        """Write a human-readable Markdown schema document."""
        lines = [
            f"# Schema: {self.symbol}  —  v{self.version}",
            "",
            f"**Pipeline version:** {self.pipeline_version}  ",
            f"**Created:** {self.created_at}  ",
            f"**Frozen:** {self.frozen}  ",
            f"**Features:** {self.feature_count}  ",
            f"**Hash:** `{self.schema_hash[:16]}…`",
            "",
            "## Feature Definitions",
            "",
            "| Name | Prefix | Category | Dtype | Nullable | Min | Max | Description |",
            "|------|--------|----------|-------|----------|-----|-----|-------------|",
        ]
        for f in self.features.values():
            lines.append(
                f"| `{f.name}` | `{f.prefix}` | {f.category} | {f.dtype} "
                f"| {f.nullable} | {f.min_value} | {f.max_value} "
                f"| {f.description[:60]} |"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
