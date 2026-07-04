"""
Window Metadata
===============
Dataclass + JSON serialization for a single walk-forward window result.

Each window is saved as ``data/ml/windows/window_{N:03d}/metadata.json``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_TS_FMT = "%Y-%m-%dT%H:%M:%S"


@dataclass
class SplitStats:
    start:     str
    end:       str
    row_count: int
    duration_days: float


@dataclass
class WindowMeta:
    """All metadata for one walk-forward window."""
    window_number:  int
    window_type:    str
    schema_version: str

    # Split boundaries
    train: SplitStats
    val:   SplitStats
    test:  SplitStats

    # Config echoed back
    train_period:   str
    val_period:     str
    test_period:    str
    step_period:    str
    gap_bars:       int

    # Validation
    validation_passed: bool
    validation_issues: list[str] = field(default_factory=list)

    # Artefact paths (relative to project root)
    artefact_paths: dict[str, str] = field(default_factory=dict)

    # Dataset info
    feature_count:  int = 0
    label_count:    int = 0
    total_columns:  int = 0

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        logger.debug("Window metadata saved → %s", path)

    @classmethod
    def from_json(cls, path: Path) -> "WindowMeta":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            window_number   = data["window_number"],
            window_type     = data["window_type"],
            schema_version  = data["schema_version"],
            train = SplitStats(**data["train"]),
            val   = SplitStats(**data["val"]),
            test  = SplitStats(**data["test"]),
            train_period    = data["train_period"],
            val_period      = data["val_period"],
            test_period     = data["test_period"],
            step_period     = data["step_period"],
            gap_bars        = data["gap_bars"],
            validation_passed = data["validation_passed"],
            validation_issues = data.get("validation_issues", []),
            artefact_paths  = data.get("artefact_paths", {}),
            feature_count   = data.get("feature_count", 0),
            label_count     = data.get("label_count", 0),
            total_columns   = data.get("total_columns", 0),
        )

    @classmethod
    def build(
        cls,
        window_number:     int,
        window_type:       str,
        train_df:          pd.DataFrame,
        val_df:            pd.DataFrame,
        test_df:           pd.DataFrame,
        train_period:      str,
        val_period:        str,
        test_period:       str,
        step_period:       str,
        gap_bars:          int,
        validation_passed: bool,
        validation_issues: list[str],
        artefact_paths:    dict[str, str],
        feature_cols:      list[str],
        label_cols:        list[str],
        schema_version:    str = "1.0.0",
    ) -> "WindowMeta":
        def _stats(df: pd.DataFrame) -> SplitStats:
            if df.empty:
                return SplitStats(start="", end="", row_count=0, duration_days=0.0)
            start = df.index[0].strftime(_TS_FMT)
            end   = df.index[-1].strftime(_TS_FMT)
            days  = (df.index[-1] - df.index[0]).total_seconds() / 86_400
            return SplitStats(start=start, end=end, row_count=len(df), duration_days=round(days, 2))

        return cls(
            window_number   = window_number,
            window_type     = window_type,
            schema_version  = schema_version,
            train           = _stats(train_df),
            val             = _stats(val_df),
            test            = _stats(test_df),
            train_period    = train_period,
            val_period      = val_period,
            test_period     = test_period,
            step_period     = step_period,
            gap_bars        = gap_bars,
            validation_passed = validation_passed,
            validation_issues = validation_issues,
            artefact_paths  = artefact_paths,
            feature_count   = len(feature_cols),
            label_count     = len(label_cols),
            total_columns   = len(feature_cols) + len(label_cols),
        )
