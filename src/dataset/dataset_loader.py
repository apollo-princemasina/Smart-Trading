"""
Dataset Loader
==============
Read-only module responsible for loading features and labels from their
respective stores and filtering them to the requested subset.

Feature sources
---------------
1. Feature Store API  (``src.feature_store.FeatureStore``)
2. Direct parquet path (for testing or ad-hoc use)

Label sources
-------------
1. Label Store directory (``data/labels/{symbol}/labels_{symbol}_v*.parquet``)
2. Direct parquet path

Feature sets
------------
top25 / top50 / top75 / top100 / top150
    Load from ``reports/selected_features_top{N}.json``
all
    Return all columns from the Feature Store
custom
    Use the caller-supplied ``custom_features`` list

Label groups
------------
market_bias       fwd_return_*, direction_*, bias_*, confidence_*, probability_*
setup_quality     setup_*
entry_timing      entry_*, is_optimal_*, optimal_entry_*, time_to_*
trade_outcome     long_*, short_*, outcome*, mfe_*, mae_*, realized_rr, expected_*, trade_duration_*
trade_management  mgmt_*
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Map label group names → column prefix tuples
LABEL_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "market_bias": (
        "fwd_return_", "direction_", "bias_", "confidence_", "probability_",
    ),
    "setup_quality": ("setup_",),
    "entry_timing": (
        "entry_", "is_optimal_", "optimal_entry_", "time_to_",
    ),
    "trade_outcome": (
        "long_", "short_", "outcome", "mfe_", "mae_", "realized_rr",
        "expected_", "trade_duration_",
    ),
    "trade_management": ("mgmt_",),
}

_FEATURE_SET_MAP = {
    "top25":  "selected_features_top25.json",
    "top50":  "selected_features_top50.json",
    "top75":  "selected_features_top75.json",
    "top100": "selected_features_top100.json",
    "top150": "selected_features_top150.json",
    "all":    None,   # load all columns
    "custom": None,   # caller supplies list
}


class DatasetLoader:
    """Load and filter features + labels for dataset assembly."""

    def __init__(
        self,
        feature_store_dir:   Optional[Path] = None,
        schema_dir:          Optional[Path] = None,
        label_dir:           Optional[Path] = None,
        feature_quality_dir: Optional[Path] = None,
    ) -> None:
        from config.settings import (
            FEATURE_STORE_DIR, SCHEMA_DIR, LABEL_DIR, QUALITY_REPORT_DIR,
        )
        self.feature_store_dir   = Path(feature_store_dir or FEATURE_STORE_DIR)
        self.schema_dir          = Path(schema_dir or SCHEMA_DIR)
        self.label_dir           = Path(label_dir or LABEL_DIR)
        self.feature_quality_dir = Path(feature_quality_dir or QUALITY_REPORT_DIR)

    # ── Features ─────────────────────────────────────────────────────────

    def load_features(
        self,
        symbol:          str,
        feature_set:     str            = "all",
        custom_features: Optional[list[str]] = None,
        version:         Optional[int]  = None,
        parquet_path:    Optional[Path] = None,
    ) -> pd.DataFrame:
        """Load feature DataFrame, optionally filtered to a selected set.

        Args:
            symbol:          Instrument identifier.
            feature_set:     One of top25/top50/top75/top100/top150/all/custom.
            custom_features: Column list (used when feature_set == 'custom').
            version:         Feature Store version (None = latest).
            parquet_path:    Direct parquet path override (bypasses Feature Store).

        Returns:
            Feature DataFrame with DatetimeIndex, sorted ascending.
        """
        if parquet_path is not None:
            df = pd.read_parquet(parquet_path)
        else:
            df = self._load_from_feature_store(symbol, version)

        df = df.sort_index()
        return self._filter_feature_set(df, feature_set, custom_features)

    def _load_from_feature_store(self, symbol: str, version: Optional[int]) -> pd.DataFrame:
        try:
            from src.feature_store import FeatureStore
            fs = FeatureStore(self.feature_store_dir, self.schema_dir)
            if version is not None:
                return fs.load_version(symbol, version)
            return fs.load_latest(symbol)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load features for '{symbol}' from Feature Store: {exc}"
            ) from exc

    def _filter_feature_set(
        self,
        df:              pd.DataFrame,
        feature_set:     str,
        custom_features: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if feature_set == "all":
            return df
        if feature_set == "custom":
            if not custom_features:
                raise ValueError("feature_set='custom' requires non-empty custom_features list.")
            available = [c for c in custom_features if c in df.columns]
            missing   = set(custom_features) - set(available)
            if missing:
                logger.warning("Custom features not found (%d): %s…", len(missing), list(missing)[:3])
            return df[available]

        feat_list = self.get_feature_list(feature_set)
        if not feat_list:
            logger.warning("Feature list for '%s' is empty; returning all features.", feature_set)
            return df
        available = [c for c in feat_list if c in df.columns]
        missing   = set(feat_list) - set(available)
        if missing:
            logger.warning(
                "Selected feature set '%s': %d/%d features not in DataFrame.",
                feature_set, len(missing), len(feat_list),
            )
        return df[available] if available else df

    def get_feature_list(self, feature_set: str) -> list[str]:
        """Return the list of feature names for a named feature set.

        Reads from the feature-quality report JSON files.
        """
        filename = _FEATURE_SET_MAP.get(feature_set)
        if filename is None:
            return []
        path = self.feature_quality_dir / filename
        if not path.exists():
            logger.warning("Feature set JSON not found: %s", path)
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Handle both ["f1","f2",...] and {"features":["f1","f2",...]}
        if isinstance(raw, list):
            return [str(f) for f in raw]
        if isinstance(raw, dict):
            for key in ("features", "selected", "columns"):
                if key in raw:
                    return [str(f) for f in raw[key]]
        logger.warning("Unexpected JSON format in %s", path)
        return []

    # ── Labels ────────────────────────────────────────────────────────────

    def load_labels(
        self,
        symbol:        str,
        label_groups:  Optional[list[str]] = None,
        custom_labels: Optional[list[str]] = None,
        version:       Optional[int]       = None,
        parquet_path:  Optional[Path]      = None,
    ) -> pd.DataFrame:
        """Load label DataFrame, optionally filtered to selected groups.

        Args:
            symbol:       Instrument identifier.
            label_groups: List of group names (None = all groups).
            custom_labels: Explicit column list override.
            version:      Label version (None = latest).
            parquet_path: Direct parquet path override.

        Returns:
            Label DataFrame with DatetimeIndex, sorted ascending.
        """
        if parquet_path is not None:
            df = pd.read_parquet(parquet_path)
        else:
            df = self._load_label_file(symbol, version)

        df = df.sort_index()
        return self._filter_label_groups(df, label_groups, custom_labels)

    def _load_label_file(self, symbol: str, version: Optional[int]) -> pd.DataFrame:
        sym_dir = self.label_dir / symbol
        if not sym_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {sym_dir}. "
                "Run LabelPipeline.run() first."
            )
        if version is not None:
            path = sym_dir / f"labels_{symbol}_v{version}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Label file not found: {path}")
        else:
            # Find the latest version
            files = sorted(sym_dir.glob(f"labels_{symbol}_v*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"No label parquet files found in {sym_dir}."
                )
            path = files[-1]
        logger.info("Loading labels: %s", path)
        return pd.read_parquet(path)

    def _filter_label_groups(
        self,
        df:            pd.DataFrame,
        label_groups:  Optional[list[str]],
        custom_labels: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if custom_labels:
            available = [c for c in custom_labels if c in df.columns]
            missing   = set(custom_labels) - set(available)
            if missing:
                logger.warning("Custom labels not found: %s", list(missing)[:3])
            return df[available]

        if label_groups is None:
            return df   # all columns

        selected: list[str] = []
        for group in label_groups:
            prefixes = LABEL_GROUP_PREFIXES.get(group)
            if prefixes is None:
                valid_groups = list(LABEL_GROUP_PREFIXES.keys())
                raise ValueError(
                    f"Unknown label group '{group}'. "
                    f"Valid groups: {valid_groups}"
                )
            for col in df.columns:
                if any(col.startswith(p) for p in prefixes) and col not in selected:
                    selected.append(col)

        if not selected:
            logger.warning(
                "No columns matched label groups %s. Returning all labels.", label_groups
            )
            return df
        return df[selected]

    # ── Utilities ─────────────────────────────────────────────────────────

    def list_label_versions(self, symbol: str) -> list[int]:
        """Return sorted list of available label versions for a symbol."""
        sym_dir = self.label_dir / symbol
        if not sym_dir.exists():
            return []
        pattern = re.compile(rf"labels_{re.escape(symbol)}_v(\d+)\.parquet")
        versions = []
        for f in sym_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                versions.append(int(m.group(1)))
        return sorted(versions)

    def list_available_feature_sets(self) -> list[str]:
        """Return feature set names that have a corresponding JSON file."""
        available = []
        for name, filename in _FEATURE_SET_MAP.items():
            if filename is None:
                available.append(name)
                continue
            if (self.feature_quality_dir / filename).exists():
                available.append(name)
        return available
