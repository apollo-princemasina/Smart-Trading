"""Feature catalog — index, search, and documentation generator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .exceptions import CatalogError
from .schema import DatasetSchema, FeatureSchema

logger = logging.getLogger(__name__)


class FeatureCatalog:
    """
    Searchable index of all features registered in a :class:`DatasetSchema`.

    Usage
    -----
    ::

        catalog = FeatureCatalog()
        catalog.build(schema)
        results = catalog.search(category="momentum", prefix="h1_")
        catalog.to_json(Path("feature_catalog.json"))
        catalog.to_markdown(Path("feature_catalog.md"))
    """

    def __init__(self):
        self._schema: DatasetSchema | None = None
        self._index:  list[FeatureSchema]  = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, schema: DatasetSchema) -> "FeatureCatalog":
        """Index all features in *schema*."""
        self._schema = schema
        self._index  = list(schema.features.values())
        logger.info(
            "Catalog built for %s v%s — %d features",
            schema.symbol, schema.version, len(self._index)
        )
        return self

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        category:     str | None = None,
        prefix:       str | None = None,
        timeframe:    str | None = None,
        source_module: str | None = None,
        tag:          str | None = None,
        deprecated:   bool | None = None,
        query:        str | None = None,   # substring match on name/description
    ) -> list[FeatureSchema]:
        """
        Return features matching all supplied filters (AND semantics).

        All arguments are optional; omitting a filter means «any value».
        """
        results = self._index[:]

        if category is not None:
            results = [f for f in results if f.category == category]
        if prefix is not None:
            results = [f for f in results if f.prefix == prefix]
        if timeframe is not None:
            results = [f for f in results if f.timeframe == timeframe]
        if source_module is not None:
            results = [f for f in results if f.source_module == source_module]
        if tag is not None:
            results = [f for f in results if tag in f.tags]
        if deprecated is not None:
            results = [f for f in results if f.deprecated == deprecated]
        if query is not None:
            q = query.lower()
            results = [
                f for f in results
                if q in f.name.lower() or q in f.description.lower()
            ]
        return results

    def get(self, feature_name: str) -> FeatureSchema | None:
        """Return the schema for an exact *feature_name*, or None."""
        if self._schema is None:
            return None
        return self._schema.features.get(feature_name)

    # ── Listing helpers ───────────────────────────────────────────────────────

    def list_categories(self) -> list[str]:
        return sorted({f.category for f in self._index if f.category})

    def list_prefixes(self) -> list[str]:
        return sorted({f.prefix for f in self._index if f.prefix})

    def list_timeframes(self) -> list[str]:
        return sorted({f.timeframe for f in self._index if f.timeframe})

    def list_source_modules(self) -> list[str]:
        return sorted({f.source_module for f in self._index if f.source_module})

    def list_tags(self) -> list[str]:
        tags: set[str] = set()
        for f in self._index:
            tags.update(f.tags)
        return sorted(tags)

    @property
    def feature_count(self) -> int:
        return len(self._index)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        if self._schema is None:
            raise CatalogError("Catalog not built — call build() first")
        return {
            "symbol":        self._schema.symbol,
            "schema_version": self._schema.version,
            "feature_count": self.feature_count,
            "categories":    self.list_categories(),
            "prefixes":      self.list_prefixes(),
            "timeframes":    self.list_timeframes(),
            "features":      {f.name: f.to_dict() for f in self._index},
        }

    def to_json(self, path: Path, indent: int = 2) -> None:
        """Write the catalog as ``feature_catalog.json``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=indent, default=str),
            encoding="utf-8",
        )
        logger.info("Wrote feature catalog → %s", path)

    def to_markdown(self, path: Path) -> None:
        """Write a Markdown feature catalog document."""
        if self._schema is None:
            raise CatalogError("Catalog not built — call build() first")

        lines = [
            f"# Feature Catalog — {self._schema.symbol}  v{self._schema.version}",
            "",
            f"**Total features:** {self.feature_count}  ",
            f"**Categories:** {', '.join(self.list_categories())}  ",
            f"**Prefixes:** {', '.join(self.list_prefixes())}  ",
            "",
        ]

        # Group by category
        by_cat: dict[str, list[FeatureSchema]] = {}
        for f in self._index:
            by_cat.setdefault(f.category or "other", []).append(f)

        for cat in sorted(by_cat):
            lines += [
                f"## {cat.replace('_', ' ').title()}",
                "",
                "| Feature | Prefix | Dtype | Nullable | Source | Description |",
                "|---------|--------|-------|----------|--------|-------------|",
            ]
            for f in sorted(by_cat[cat], key=lambda x: x.name):
                dep = f.deprecated and " ⚠️ _deprecated_" or ""
                lines.append(
                    f"| `{f.name}` | `{f.prefix}` | {f.dtype} | {f.nullable} "
                    f"| {f.source_module} | {f.description[:70]}{dep} |"
                )
            lines.append("")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote feature catalog markdown → %s", path)

    # ── Class-level builder ───────────────────────────────────────────────────

    @classmethod
    def from_schema(cls, schema: DatasetSchema) -> "FeatureCatalog":
        """Convenience: build and return a catalog from *schema* in one call."""
        return cls().build(schema)

    @classmethod
    def from_json(cls, path: Path) -> "FeatureCatalog":
        """Reconstruct a catalog index from a saved JSON file."""
        raw  = json.loads(path.read_text(encoding="utf-8"))
        cat  = cls()
        cat._index = [
            FeatureSchema.from_dict(v) for v in raw.get("features", {}).values()
        ]
        return cat
