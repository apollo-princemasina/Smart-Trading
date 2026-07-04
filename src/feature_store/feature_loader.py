"""Low-level Parquet I/O with dataset versioning and manifest tracking."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from .dataset_manifest import DatasetManifest, ManifestIndex
from .exceptions import DatasetNotFoundError
from .schema import DatasetSchema

logger = logging.getLogger(__name__)

_DATASET_STEM = "feature_dataset_v{n}"
_DATASET_EXT  = ".parquet"


class FeatureLoader:
    """
    Saves and loads versioned Parquet datasets with manifest tracking.

    Dataset files
    -------------
    Each dataset version is stored as a separate file — old versions are
    never overwritten::

        {base_dir}/{symbol}/feature_dataset_v1.parquet
        {base_dir}/{symbol}/feature_dataset_v2.parquet
        …

    Every save operation also writes ``manifest_v{N}.json`` and updates
    ``manifest_index.json``.
    """

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _symbol_dir(self, symbol: str) -> Path:
        return self._base / symbol

    def _dataset_path(self, symbol: str, version: int) -> Path:
        return self._symbol_dir(symbol) / f"{_DATASET_STEM.format(n=version)}{_DATASET_EXT}"

    def _manifest_path(self, symbol: str, version: int) -> Path:
        return self._symbol_dir(symbol) / f"manifest_v{version}.json"

    def _hash_file(self, symbol: str) -> Path:
        return self._symbol_dir(symbol) / "schema.hash"

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        schema: DatasetSchema,
    ) -> tuple[Path, int]:
        """
        Save *df* as the next Parquet version for *symbol*.

        Returns
        -------
        ``(saved_path, version_number)``
        """
        sym_dir = self._symbol_dir(symbol)
        sym_dir.mkdir(parents=True, exist_ok=True)

        idx     = ManifestIndex(sym_dir)
        version = idx.next_version()
        path    = self._dataset_path(symbol, version)

        # Write Parquet (index=True preserves DatetimeIndex)
        df.to_parquet(path, engine="pyarrow", index=True)
        logger.info(
            "Saved dataset v%d for %s → %s (%d rows × %d cols)",
            version, symbol, path, len(df), df.shape[1],
        )

        # Compute file hash
        sha256 = _file_hash(path)

        # Write manifest
        manifest = DatasetManifest.generate(df, schema, version, path)
        manifest.sha256_hash = sha256
        manifest.save(self._manifest_path(symbol, version))
        idx.register(manifest)

        # Write schema hash file (used for training/inference verification)
        if schema.schema_hash:
            self._hash_file(symbol).write_text(schema.schema_hash, encoding="utf-8")

        return path, version

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_version(self, symbol: str, version: int) -> pd.DataFrame:
        """Load dataset version *version* for *symbol*."""
        path = self._dataset_path(symbol, version)
        if not path.exists():
            raise DatasetNotFoundError(
                f"Dataset not found: symbol={symbol!r}, version={version}"
            )
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(
            "Loaded dataset v%d for %s (%d rows × %d cols)",
            version, symbol, len(df), df.shape[1],
        )
        return df

    def load_latest(self, symbol: str) -> pd.DataFrame:
        """Load the most recently saved dataset for *symbol*."""
        idx     = ManifestIndex(self._symbol_dir(symbol))
        latest  = idx.latest_version()
        if latest is None:
            raise DatasetNotFoundError(
                f"No datasets saved for symbol {symbol!r}"
            )
        return self.load_version(symbol, latest)

    def load_subset(
        self,
        symbol: str,
        columns: list[str],
        version: int | None = None,
    ) -> pd.DataFrame:
        """Load only *columns* from the dataset (Parquet column pruning)."""
        sym_dir = self._symbol_dir(symbol)
        if version is None:
            idx     = ManifestIndex(sym_dir)
            version = idx.latest_version()
            if version is None:
                raise DatasetNotFoundError(f"No datasets for {symbol!r}")
        path = self._dataset_path(symbol, version)
        if not path.exists():
            raise DatasetNotFoundError(
                f"Dataset v{version} not found for {symbol!r}"
            )
        return pd.read_parquet(path, engine="pyarrow", columns=columns)

    # ── Manifest access ───────────────────────────────────────────────────────

    def list_versions(self, symbol: str) -> list[int]:
        """Return all saved dataset version numbers (ascending)."""
        return ManifestIndex(self._symbol_dir(symbol)).versions()

    def get_manifest(self, symbol: str, version: int) -> DatasetManifest:
        path = self._manifest_path(symbol, version)
        return DatasetManifest.load(path)

    def list_manifests(self, symbol: str) -> list[DatasetManifest]:
        """Return all manifests for *symbol*, ordered by version."""
        versions = self.list_versions(symbol)
        return [self.get_manifest(symbol, v) for v in versions]

    # ── Archive ───────────────────────────────────────────────────────────────

    def archive(self, symbol: str, version: int) -> Path:
        """
        Move dataset version *version* to an ``_archive`` subdirectory.

        The dataset is still available but will not appear in :meth:`list_versions`.
        """
        src     = self._dataset_path(symbol, version)
        archive = self._symbol_dir(symbol) / "_archive"
        archive.mkdir(parents=True, exist_ok=True)
        dst     = archive / src.name
        src.rename(dst)
        logger.info("Archived dataset v%d for %s → %s", version, symbol, dst)
        return dst

    # ── Saved schema hash access ──────────────────────────────────────────────

    def load_schema_hash(self, symbol: str) -> str | None:
        """Return the last saved schema hash for *symbol*, or None."""
        p = self._hash_file(symbol)
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    # ── Listing ───────────────────────────────────────────────────────────────

    def list_symbols(self) -> list[str]:
        """Return all symbols that have at least one saved dataset."""
        return sorted(
            d.name
            for d in self._base.iterdir()
            if d.is_dir() and any(d.glob("feature_dataset_v*.parquet"))
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _file_hash(path: Path) -> str:
    """SHA-256 hash of a file's content."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()
