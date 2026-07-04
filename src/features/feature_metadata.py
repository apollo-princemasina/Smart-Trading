"""FeatureMetadata — structured descriptor attached to every feature generator.

Metadata serves three purposes:
1. Human documentation (what this feature does, who wrote it, when).
2. Pipeline introspection (dependencies, required input columns, output columns).
3. ML explainability (category and complexity help SHAP post-processing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class FeatureMetadata:
    """Describes a single feature generator registered in the pipeline.

    Attributes
    ----------
    name:
        Unique registry key. Must match ``BaseFeature.name``.
    category:
        Logical group (market_structure, trend, volatility, …).
    description:
        One-sentence human-readable summary of what this feature computes.
    dependencies:
        Names of other features that must run *before* this one.
        The registry uses this list to sort execution order.
    required_columns:
        Columns from the input OHLCV (merged M15) DataFrame that this
        generator reads.  The pipeline validates these are present before
        calling ``generate()``.
    output_columns:
        Column names that ``generate()`` will add to the dataset.
        Populated after the first successful execution.
    version:
        Semantic version of the feature implementation.
    author:
        Name or team responsible for maintaining this generator.
    created_at:
        ISO-8601 date string (YYYY-MM-DD) when the feature was first written.
    updated_at:
        ISO-8601 date string updated on each implementation change.
    complexity:
        Rough compute cost: ``"low"`` / ``"medium"`` / ``"high"``.
        Used to estimate pipeline execution budgets.
    execution_time_ms:
        Wall-clock milliseconds for the last ``generate()`` call.
        Updated by the pipeline after each run.
    enabled:
        If ``False`` the registry skips this feature entirely.
    tags:
        Free-form labels for filtering (e.g. ``["ICT", "smart_money"]``).
    """

    name: str
    category: str
    description: str

    dependencies: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)

    version: str = "1.0.0"
    author: str = "Smart Trading Team"
    created_at: str = field(default_factory=_today)
    updated_at: str = field(default_factory=_today)

    complexity: str = "low"          # "low" | "medium" | "high"
    execution_time_ms: float = 0.0
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON / Markdown reporting."""
        return {
            "name":               self.name,
            "category":           self.category,
            "description":        self.description,
            "dependencies":       self.dependencies,
            "required_columns":   self.required_columns,
            "output_columns":     self.output_columns,
            "version":            self.version,
            "author":             self.author,
            "created_at":         self.created_at,
            "updated_at":         self.updated_at,
            "complexity":         self.complexity,
            "execution_time_ms":  self.execution_time_ms,
            "enabled":            self.enabled,
            "tags":               self.tags,
        }

    def __repr__(self) -> str:
        return (
            f"FeatureMetadata(name={self.name!r}, "
            f"category={self.category!r}, "
            f"complexity={self.complexity!r})"
        )
