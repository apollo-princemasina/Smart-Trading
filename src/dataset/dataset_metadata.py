"""
Dataset Metadata
================
Captures and serialises all provenance information for a training dataset:

  - Identification: name, version, creation timestamp
  - Source: symbol, timeframe, feature version, label version
  - Dimensions: row count, feature count, label count, date range
  - Quality: missing value summary, class distributions
  - Lineage: pipeline version, schema version, feature set, label groups
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


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    if isinstance(obj, Path):           return str(obj)
    raise TypeError(f"Not JSON-serialisable: {type(obj)}")


@dataclass
class ColumnSummary:
    name:        str
    dtype:       str
    nan_count:   int
    nan_rate:    float
    n_valid:     int
    is_label:    bool
    value_range: dict    # {min, max} or {classes: [...]}


@dataclass
class DatasetMeta:
    # Identification
    dataset_name:      str
    dataset_version:   int
    schema_version:    str
    label_version:     int
    pipeline_version:  str
    created_at:        str           # ISO-8601 UTC

    # Source
    symbol:            str
    prediction_timeframe: str
    higher_timeframes: list[str]
    feature_set:       str
    label_groups:      list[str]

    # Dimensions
    row_count:         int
    feature_count:     int
    label_count:       int
    column_count:      int
    start_date:        str
    end_date:          str

    # Quality
    total_missing:     int
    missing_rate:      float
    column_summaries:  list[ColumnSummary]
    class_distributions: dict         # {col: {class: count}}

    # Lineage
    feature_columns:   list[str]
    label_columns:     list[str]
    validation_passed: bool
    validation_summary: str
    artefact_paths:    dict

    notes: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Optional[Path] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), default=_json_default, indent=indent)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
            logger.info("DatasetMeta saved → %s", path)
        return text

    @classmethod
    def from_json(cls, path: Path) -> "DatasetMeta":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["column_summaries"] = [ColumnSummary(**c) for c in data["column_summaries"]]
        return cls(**data)

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        dataset:           pd.DataFrame,
        feature_columns:   list[str],
        label_columns:     list[str],
        symbol:            str,
        dataset_name:      str           = "training_dataset",
        dataset_version:   int           = 1,
        schema_version:    str           = "1.0.0",
        label_version:     int           = 1,
        pipeline_version:  str           = "1.0.0",
        prediction_timeframe: str        = "",
        higher_timeframes: list[str]     = None,
        feature_set:       str           = "custom",
        label_groups:      list[str]     = None,
        validation_passed: bool          = True,
        validation_summary: str          = "",
        artefact_paths:    dict          = None,
        notes:             str           = "",
    ) -> "DatasetMeta":
        now      = datetime.now(timezone.utc).isoformat()
        n_rows   = len(dataset)
        idx      = dataset.index

        start = str(idx.min()) if len(idx) else ""
        end   = str(idx.max()) if len(idx) else ""

        col_sums: list[ColumnSummary] = []
        class_dists: dict             = {}

        for col in dataset.columns:
            s     = dataset[col]
            valid = s.dropna()
            n_v   = len(valid)
            nan_c = int(s.isna().sum())
            nan_r = float(s.isna().mean())
            is_lbl = col in label_columns

            _is_numeric = pd.api.types.is_numeric_dtype(s)
            if n_v > 0 and _is_numeric and valid.nunique() <= 12:
                try:
                    vrange = {"classes": [float(v) for v in sorted(valid.unique())]}
                    class_dists[col] = {
                        str(int(k) if float(k) == int(float(k)) else k): int(v)
                        for k, v in valid.value_counts().items()
                    }
                except (TypeError, ValueError):
                    vrange = {}
            elif n_v > 0 and _is_numeric:
                try:
                    vrange = {"min": float(valid.min()), "max": float(valid.max())}
                except (TypeError, ValueError):
                    vrange = {}
            elif n_v > 0:
                # Non-numeric (datetime, string, etc.) — store as string representation
                try:
                    vrange = {"min": str(valid.min()), "max": str(valid.max())}
                except Exception:
                    vrange = {}
            else:
                vrange = {}

            col_sums.append(ColumnSummary(
                name=col, dtype=str(s.dtype),
                nan_count=nan_c, nan_rate=nan_r, n_valid=n_v,
                is_label=is_lbl, value_range=vrange,
            ))

        total_missing = int(dataset.isna().sum().sum())
        missing_rate  = float(dataset.isna().mean().mean())

        return cls(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            schema_version=schema_version,
            label_version=label_version,
            pipeline_version=pipeline_version,
            created_at=now,
            symbol=symbol,
            prediction_timeframe=prediction_timeframe,
            higher_timeframes=higher_timeframes or [],
            feature_set=feature_set,
            label_groups=label_groups or [],
            row_count=n_rows,
            feature_count=len(feature_columns),
            label_count=len(label_columns),
            column_count=len(dataset.columns),
            start_date=start,
            end_date=end,
            total_missing=total_missing,
            missing_rate=missing_rate,
            column_summaries=col_sums,
            class_distributions=class_dists,
            feature_columns=feature_columns,
            label_columns=label_columns,
            validation_passed=validation_passed,
            validation_summary=validation_summary,
            artefact_paths=artefact_paths or {},
            notes=notes,
        )
