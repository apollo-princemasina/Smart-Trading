"""Schema lifecycle orchestration — creation, freezing, evolution, rollback."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset_hash import compute_features_hash
from .exceptions import SchemaFrozenError
from .feature_prefix import extract_prefix
from .schema import DatasetSchema, FeatureSchema
from .schema_registry import SchemaRegistry
from .schema_versioning import determine_version_bump, next_version

logger = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")


class SchemaManager:
    """
    Orchestrates the full lifecycle of a :class:`DatasetSchema`.

    Responsibilities
    ----------------
    * Infer a schema from a raw DataFrame (auto-discovery).
    * Freeze schemas (immutable after publication).
    * Register new versions when the feature set evolves.
    * Load, compare, and roll back schemas via the registry.
    """

    def __init__(
        self,
        registry: SchemaRegistry,
        pipeline_version: str = "1.0.0",
        author: str = "SmartTrading",
    ):
        self._registry        = registry
        self._pipeline_version = pipeline_version
        self._author          = author

    # ── Schema creation ───────────────────────────────────────────────────────

    def infer_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        version: str = "1.0.0",
        description: str = "",
        feature_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> DatasetSchema:
        """
        Auto-infer a :class:`DatasetSchema` from *df* column dtypes.

        For each column the prefix, dtype, and nullability are derived
        automatically.  Extra per-feature metadata can be injected via
        *feature_metadata* ``{col_name: {field: value, …}}``.

        The resulting schema is **not** automatically registered; call
        :meth:`register_and_freeze` explicitly.
        """
        now            = _NOW()
        feature_meta   = feature_metadata or {}
        features: dict[str, FeatureSchema] = {}

        for col in df.columns:
            meta  = feature_meta.get(col, {})
            dtype = str(df[col].dtype)

            features[col] = FeatureSchema(
                name          = col,
                prefix        = extract_prefix(col),
                category      = meta.get("category", _guess_category(col)),
                description   = meta.get("description", ""),
                source_module = meta.get("source_module", ""),
                timeframe     = meta.get("timeframe", _guess_timeframe(col)),
                dependencies  = meta.get("dependencies", []),
                dtype         = meta.get("dtype", dtype),
                nullable      = meta.get("nullable", bool(df[col].isna().any())),
                default_value = meta.get("default_value", None),
                min_value     = meta.get("min_value", None),
                max_value     = meta.get("max_value", None),
                units         = meta.get("units", ""),
                example_value = meta.get("example_value", None),
                version       = meta.get("version", "1.0.0"),
                author        = meta.get("author", self._author),
                created_date  = now,
                updated_date  = now,
                deprecated    = meta.get("deprecated", False),
                tags          = meta.get("tags", []),
            )

        schema_hash = compute_features_hash(features)

        return DatasetSchema(
            version          = version,
            symbol           = symbol,
            features         = features,
            created_at       = now,
            frozen           = False,
            schema_hash      = schema_hash,
            pipeline_version = self._pipeline_version,
            description      = description,
        )

    def freeze(self, schema: DatasetSchema) -> DatasetSchema:
        """
        Mark *schema* as frozen and register it.

        After freezing the schema may not be mutated.  Returns a new
        :class:`DatasetSchema` instance with ``frozen=True``.
        """
        if schema.frozen:
            logger.debug("Schema %s v%s already frozen", schema.symbol, schema.version)
            return schema

        frozen = DatasetSchema(
            version          = schema.version,
            symbol           = schema.symbol,
            features         = schema.features,
            created_at       = schema.created_at,
            frozen           = True,
            schema_hash      = schema.schema_hash,
            pipeline_version = schema.pipeline_version,
            description      = schema.description,
        )
        self._registry.register(frozen)
        logger.info("Frozen schema %s v%s", frozen.symbol, frozen.version)
        return frozen

    def register_and_freeze(
        self,
        df: pd.DataFrame,
        symbol: str,
        version: str = "1.0.0",
        **kwargs,
    ) -> DatasetSchema:
        """Convenience: infer, freeze, and register in one call."""
        schema = self.infer_from_dataframe(df, symbol, version, **kwargs)
        return self.freeze(schema)

    # ── Schema evolution ──────────────────────────────────────────────────────

    def evolve(
        self,
        current: DatasetSchema,
        new_features: dict[str, FeatureSchema],
        force_version: str | None = None,
    ) -> DatasetSchema:
        """
        Produce a new :class:`DatasetSchema` from *current* + updated features.

        The version is bumped automatically (MAJOR / MINOR / PATCH) unless
        *force_version* is supplied.  The new schema is **not** registered
        automatically.
        """
        if current.frozen and not force_version:
            pass   # frozen schemas can still be evolved — just creates a new version

        new_ver = force_version or next_version(
            current.version, current.features, new_features
        )
        schema_hash = compute_features_hash(new_features)
        now = _NOW()

        evolved = DatasetSchema(
            version          = new_ver,
            symbol           = current.symbol,
            features         = new_features,
            created_at       = now,
            frozen           = False,
            schema_hash      = schema_hash,
            pipeline_version = current.pipeline_version,
            description      = current.description,
        )
        logger.info(
            "Evolved schema %s: %s → %s (%s bump)",
            current.symbol,
            current.version,
            new_ver,
            determine_version_bump(current.features, new_features),
        )
        return evolved

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, symbol: str, version: str = "latest") -> DatasetSchema:
        """Load a schema by symbol and version string (or ``"latest"``)."""
        if version == "latest":
            return self._registry.get_latest(symbol)
        return self._registry.get(symbol, version)

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare(self, schema_a: DatasetSchema, schema_b: DatasetSchema) -> dict:
        """Return a structured diff between two schemas."""
        a_feats = set(schema_a.features)
        b_feats = set(schema_b.features)

        added   = sorted(b_feats - a_feats)
        removed = sorted(a_feats - b_feats)

        type_changes: dict[str, dict] = {}
        for name in a_feats & b_feats:
            fa, fb = schema_a.features[name], schema_b.features[name]
            if fa.dtype != fb.dtype:
                type_changes[name] = {"from": fa.dtype, "to": fb.dtype}

        bump = determine_version_bump(schema_a.features, schema_b.features)

        return {
            "symbol":        schema_a.symbol,
            "version_a":     schema_a.version,
            "version_b":     schema_b.version,
            "added":         added,
            "removed":       removed,
            "type_changes":  type_changes,
            "version_bump":  bump,
            "hashes_equal":  schema_a.schema_hash == schema_b.schema_hash,
        }

    # ── Rollback ──────────────────────────────────────────────────────────────

    def rollback(self, symbol: str, to_version: str) -> DatasetSchema:
        """
        Pin *to_version* as the active schema for *symbol*.

        After this call, :meth:`load` with ``version="latest"`` returns
        the pinned version.
        """
        schema = self._registry.get(symbol, to_version)
        self._registry.pin(symbol, to_version)
        logger.warning(
            "Rolled back schema %s to v%s", symbol, to_version
        )
        return schema


# ── Inference helpers ─────────────────────────────────────────────────────────


def _guess_category(col_name: str) -> str:
    """Heuristic category guess from column prefix."""
    from .feature_prefix import extract_prefix

    prefix = extract_prefix(col_name)
    mapping = {
        "ms_":     "market_structure",
        "liq_":    "liquidity",
        "tech_":   "technical",
        "stat_":   "statistical",
        "vol_":    "volatility",
        "sess_":   "session",
        "label_":  "label",
        "future_": "label",
        "macro_":  "macro",
        "news_":   "news",
        "sent_":   "sentiment",
    }
    if prefix in mapping:
        return mapping[prefix]
    # Timeframe prefixes → infer from name suffix
    for kw in ("rsi", "macd", "ema", "sma", "bband", "stoch"):
        if kw in col_name:
            return "technical"
    for kw in ("bos", "choch", "ob", "fvg", "swing"):
        if kw in col_name:
            return "market_structure"
    return "feature"


def _guess_timeframe(col_name: str) -> str:
    """Heuristic timeframe guess from column prefix."""
    from .feature_prefix import extract_prefix

    prefix_to_tf = {
        "weekly_": "W",
        "daily_":  "D",
        "h4_":     "H4",
        "h1_":     "H1",
        "m15_":    "M15",
    }
    return prefix_to_tf.get(extract_prefix(col_name), "")
