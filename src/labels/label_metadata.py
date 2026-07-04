"""
Label Metadata
==============
Captures and serialises all metadata about a label generation run:

  - Symbol, timeframe, label version
  - Generation config for every model
  - Row counts, valid counts, NaN rates
  - Class distributions
  - Validation result summary
  - File paths of saved artefacts

The metadata is written alongside every labels_v{N}.parquet file so
that any consumer can reconstruct how the labels were produced.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _default_json(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON-serialisable")


@dataclass
class ColumnMeta:
    name:        str
    dtype:       str
    nan_rate:    float
    n_valid:     int
    value_range: dict   # {min, max} or {classes: [...]}


@dataclass
class LabelMeta:
    symbol:         str
    timeframe:      str
    label_version:  int
    generated_at:   str          # ISO-8601 UTC
    generator:      str          # e.g. "LabelPipeline v1"
    n_rows:         int
    n_valid_rows:   int          # rows with at least one non-NaN label
    label_columns:  list[str]
    column_meta:    list[ColumnMeta]
    class_distributions: dict    # {col: {class: count}}
    config_snapshot: dict        # serialised pipeline config
    validation_summary: str
    validation_passed:  bool
    artefact_paths: dict         # {name: path_str}
    notes:          str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: Optional[Path] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), default=_default_json, indent=indent)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
            logger.info("LabelMetadata saved → %s", path)
        return text

    @classmethod
    def from_json(cls, path: Path) -> "LabelMeta":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["column_meta"] = [ColumnMeta(**c) for c in data["column_meta"]]
        return cls(**data)

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        labels:             pd.DataFrame,
        symbol:             str,
        timeframe:          str    = "",
        label_version:      int    = 1,
        generator:          str    = "LabelPipeline v1",
        config_snapshot:    dict   = None,
        validation_summary: str    = "",
        validation_passed:  bool   = True,
        artefact_paths:     dict   = None,
        notes:              str    = "",
    ) -> "LabelMeta":
        """Build metadata from a label DataFrame."""
        config_snapshot = config_snapshot or {}
        artefact_paths  = artefact_paths  or {}

        n_rows      = len(labels)
        n_valid     = int(labels.notna().any(axis=1).sum())
        label_cols  = list(labels.columns)

        column_meta: list[ColumnMeta] = []
        class_dists: dict             = {}

        for col in label_cols:
            series = labels[col]
            valid  = series.dropna()
            n_v    = len(valid)
            nan_r  = float(series.isna().mean())

            if n_v == 0:
                vrange: dict = {"min": None, "max": None}
            elif valid.nunique() <= 10 and valid.dtype.kind in ("i", "f"):
                uniq   = sorted(valid.unique().tolist())
                vrange = {"classes": [float(u) for u in uniq]}
                class_dists[col] = {
                    str(int(k) if float(k) == int(float(k)) else k): int(v)
                    for k, v in valid.value_counts().items()
                }
            else:
                vrange = {
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                }

            column_meta.append(ColumnMeta(
                name=col,
                dtype=str(series.dtype),
                nan_rate=nan_r,
                n_valid=n_v,
                value_range=vrange,
            ))

        return cls(
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            generator=generator,
            n_rows=n_rows,
            n_valid_rows=n_valid,
            label_columns=label_cols,
            column_meta=column_meta,
            class_distributions=class_dists,
            config_snapshot=config_snapshot,
            validation_summary=validation_summary,
            validation_passed=validation_passed,
            artefact_paths=artefact_paths,
            notes=notes,
        )
