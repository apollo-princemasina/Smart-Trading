"""Dataset manifest — records every saved dataset version."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .exceptions import ManifestError
from .schema import DatasetSchema

logger = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DatasetManifest:
    """
    Metadata record for a single saved dataset version.

    One manifest file (``manifest_v{N}.json``) is written alongside every
    Parquet dataset file.  A ``manifest_index.json`` file in the symbol
    directory tracks all versions.

    Example manifest
    ----------------
    ::

        {
          "dataset": "EURUSD",
          "schema_version": "1.0.0",
          "dataset_version": 1,
          "feature_count": 312,
          "rows": 198734,
          "timeframes": ["W", "D", "H4", "H1", "M15"],
          "date_start": "2017-01-01T00:00:00+00:00",
          "date_end": "2025-12-31T23:45:00+00:00",
          "pipeline_version": "1.0.0",
          "created_at": "2025-07-01T10:00:00+00:00",
          "file_path": "EURUSD/feature_dataset_v1.parquet",
          "sha256_hash": "abc123…"
        }
    """

    dataset:          str
    schema_version:   str
    dataset_version:  int
    feature_count:    int
    rows:             int
    timeframes:       list[str]
    date_start:       str
    date_end:         str
    pipeline_version: str
    created_at:       str
    file_path:        str
    sha256_hash:      str = ""
    schema_hash:      str = ""

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def generate(
        cls,
        df: pd.DataFrame,
        schema: DatasetSchema,
        dataset_version: int,
        file_path: Path,
    ) -> "DatasetManifest":
        """Build a manifest from a DataFrame, schema, and output path."""
        if not isinstance(df.index, pd.DatetimeIndex):
            date_start = str(df.index[0]) if len(df) else ""
            date_end   = str(df.index[-1]) if len(df) else ""
        else:
            date_start = df.index[0].isoformat() if len(df) else ""
            date_end   = df.index[-1].isoformat() if len(df) else ""

        timeframes = sorted(schema.timeframes) if schema.timeframes else []

        return cls(
            dataset          = schema.symbol,
            schema_version   = schema.version,
            dataset_version  = dataset_version,
            feature_count    = schema.feature_count,
            rows             = len(df),
            timeframes       = timeframes,
            date_start       = date_start,
            date_end         = date_end,
            pipeline_version = schema.pipeline_version,
            created_at       = _NOW(),
            file_path        = str(file_path),
            sha256_hash      = "",   # filled after Parquet is written
            schema_hash      = schema.schema_hash,
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetManifest":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Saved manifest v%d → %s", self.dataset_version, path)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        if not path.exists():
            raise ManifestError(f"Manifest not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── Manifest index ────────────────────────────────────────────────────────────


class ManifestIndex:
    """
    Tracks all dataset versions for one symbol via ``manifest_index.json``.

    The index maps dataset_version → manifest dict for quick lookup without
    loading every individual manifest file.
    """

    def __init__(self, symbol_dir: Path):
        self._path = symbol_dir / "manifest_index.json"
        self._data: dict[int, dict] = self._load()

    def _load(self) -> dict[int, dict]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({str(k): v for k, v in self._data.items()}, indent=2),
            encoding="utf-8",
        )

    def register(self, manifest: DatasetManifest) -> None:
        self._data[manifest.dataset_version] = manifest.to_dict()
        self._save()

    def versions(self) -> list[int]:
        return sorted(self._data)

    def latest_version(self) -> int | None:
        versions = self.versions()
        return versions[-1] if versions else None

    def next_version(self) -> int:
        lv = self.latest_version()
        return 1 if lv is None else lv + 1

    def get(self, version: int) -> dict | None:
        return self._data.get(version)

    def all(self) -> list[dict]:
        return [self._data[v] for v in self.versions()]
