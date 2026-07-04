"""Persistent schema registry — stores and retrieves DatasetSchema by version."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .exceptions import SchemaNotFoundError
from .schema import DatasetSchema
from .schema_versioning import latest_version, sort_versions

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """
    File-backed registry for :class:`DatasetSchema` objects.

    Layout on disk::

        {base_dir}/
            {symbol}/
                schema_v1.0.0.json
                schema_v1.1.0.json
                schema_v2.0.0.json
                _pinned.txt          ← optional; overrides "latest" resolution

    The ``_pinned.txt`` file holds a single version string.
    When present, :meth:`get_latest` returns that pinned version instead of
    the highest registered one (supports ``rollback``).
    """

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _symbol_dir(self, symbol: str) -> Path:
        return self._base / symbol

    def _schema_path(self, symbol: str, version: str) -> Path:
        return self._symbol_dir(symbol) / f"schema_v{version}.json"

    def _pin_path(self, symbol: str) -> Path:
        return self._symbol_dir(symbol) / "_pinned.txt"

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def register(self, schema: DatasetSchema) -> None:
        """Persist *schema* to disk.  Never overwrites an existing version."""
        path = self._schema_path(schema.symbol, schema.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            logger.debug(
                "Schema %s v%s already registered — skipping",
                schema.symbol, schema.version,
            )
            return
        schema.to_json(path)
        logger.info(
            "Registered schema %s v%s → %s", schema.symbol, schema.version, path
        )

    def get(self, symbol: str, version: str) -> DatasetSchema:
        """Load schema by exact *version* string (e.g. ``"1.0.0"``)."""
        path = self._schema_path(symbol, version)
        if not path.exists():
            raise SchemaNotFoundError(
                f"Schema not found: symbol={symbol!r}, version={version!r}"
            )
        return DatasetSchema.from_json(path)

    def get_latest(self, symbol: str) -> DatasetSchema:
        """
        Load the latest schema for *symbol*.

        If a pin is active, returns the pinned version; otherwise returns
        the schema with the highest semantic version number.
        """
        pin_path = self._pin_path(symbol)
        if pin_path.exists():
            pinned_ver = pin_path.read_text(encoding="utf-8").strip()
            logger.debug("Returning pinned schema %s v%s", symbol, pinned_ver)
            return self.get(symbol, pinned_ver)

        versions = self.list_versions(symbol)
        if not versions:
            raise SchemaNotFoundError(f"No schemas registered for symbol {symbol!r}")
        return self.get(symbol, latest_version(versions))

    def list_versions(self, symbol: str) -> list[str]:
        """Return all registered version strings, sorted ascending."""
        sym_dir = self._symbol_dir(symbol)
        if not sym_dir.exists():
            return []
        versions = [
            p.stem.replace("schema_v", "")
            for p in sym_dir.glob("schema_v*.json")
        ]
        return sort_versions(versions)

    def exists(self, symbol: str, version: str) -> bool:
        """True if the schema file for (symbol, version) exists."""
        return self._schema_path(symbol, version).exists()

    def delete(self, symbol: str, version: str) -> None:
        """Remove a schema file.  Use with caution."""
        path = self._schema_path(symbol, version)
        if path.exists():
            path.unlink()
            logger.warning("Deleted schema %s v%s", symbol, version)

    # ── Pin / rollback support ────────────────────────────────────────────────

    def pin(self, symbol: str, version: str) -> None:
        """
        Pin *version* as the active schema for *symbol*.

        After pinning, :meth:`get_latest` returns this version until
        :meth:`unpin` is called.
        """
        if not self.exists(symbol, version):
            raise SchemaNotFoundError(
                f"Cannot pin — schema {symbol!r} v{version} not registered"
            )
        pin_path = self._pin_path(symbol)
        pin_path.write_text(version, encoding="utf-8")
        logger.info("Pinned schema %s v%s", symbol, version)

    def unpin(self, symbol: str) -> None:
        """Remove any version pin — :meth:`get_latest` returns highest version."""
        pin_path = self._pin_path(symbol)
        if pin_path.exists():
            pin_path.unlink()
            logger.info("Unpinned schema %s", symbol)

    def pinned_version(self, symbol: str) -> str | None:
        """Return the currently pinned version, or None."""
        p = self._pin_path(symbol)
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    # ── Inspection ────────────────────────────────────────────────────────────

    def list_symbols(self) -> list[str]:
        """Return all symbols that have at least one registered schema."""
        return sorted(
            d.name
            for d in self._base.iterdir()
            if d.is_dir() and any(d.glob("schema_v*.json"))
        )
