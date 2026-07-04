"""BaseFeature — abstract contract every feature generator must satisfy.

Design principles
-----------------
* **Class-level attributes** (``name``, ``category``, ``dependencies``,
  ``required_columns``) are declared as ``ClassVar`` so the registry can
  inspect them on the *class* without instantiating.
* ``__init_subclass__`` enforces the contract at class-definition time so
  mistakes surface immediately, not at runtime.
* ``generate()`` and ``metadata()`` are abstract — all concrete subclasses
  must implement both.
* The built-in ``validate_output()`` method runs a minimal correctness check
  after every ``generate()`` call.  The standalone ``FeatureValidator`` adds
  deeper statistical checks on top.

Usage example
-------------
    from src.features.base_feature import BaseFeature
    from src.features.feature_registry import FeatureRegistry
    from src.features.feature_metadata import FeatureMetadata
    import pandas as pd

    @FeatureRegistry.register
    class MyTrendFeature(BaseFeature):
        name             = "my_trend"
        category         = "trend"
        dependencies     = []
        required_columns = ["close"]

        def generate(self, df: pd.DataFrame) -> pd.DataFrame:
            out = pd.DataFrame(index=df.index)
            out["my_trend_signal"] = 0.0   # placeholder
            return out

        def metadata(self) -> FeatureMetadata:
            return FeatureMetadata(
                name="my_trend",
                category="trend",
                description="Detects trend direction from price action.",
            )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from .feature_metadata import FeatureMetadata

logger = logging.getLogger(__name__)


class BaseFeature(ABC):
    """Abstract base class for all feature generators.

    Subclasses **must** define the following class-level attributes:

    name : str
        Unique, snake_case registry key.  Must be globally unique
        across the entire feature suite.
    category : str
        Logical group this feature belongs to
        (market_structure | liquidity | sessions | trend |
        volatility | momentum | volume | labels).
    dependencies : list[str]
        Names of other features whose output this generator requires.
        Declare an empty list if there are no dependencies.
    required_columns : list[str]
        Column names from the merged input DataFrame that ``generate()``
        reads.  The pipeline validates their presence before calling.

    Subclasses **must** implement:

    generate(df) -> pd.DataFrame
        Core computation.  Must return a DataFrame with the *same index*
        as the input and only the newly generated columns (no copies of
        input columns).

    metadata() -> FeatureMetadata
        Declare the feature's descriptor.  The pipeline calls this once
        to populate the feature report and ML explainability layer.
    """

    # ── Class-level contract ────────────────────────────────────────────────
    name:             ClassVar[str]        = ""
    category:         ClassVar[str]        = ""
    dependencies:     ClassVar[list[str]]  = []
    required_columns: ClassVar[list[str]]  = []

    # ── Subclass validation ─────────────────────────────────────────────────
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate that concrete subclasses declare required class attributes."""
        super().__init_subclass__(**kwargs)

        # Skip validation for intermediate abstract classes (those that still
        # have abstractmethods defined).
        if getattr(cls, "__abstractmethods__", frozenset()):
            return

        errors: list[str] = []
        if not cls.name:
            errors.append(f"{cls.__name__} must define a non-empty class attribute 'name'.")
        if not cls.category:
            errors.append(f"{cls.__name__} must define a non-empty class attribute 'category'.")

        if errors:
            raise TypeError("\n".join(errors))

    # ── Abstract interface ──────────────────────────────────────────────────
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature columns from the input OHLCV DataFrame.

        Parameters
        ----------
        df:
            The merged M15 OHLCV DataFrame produced by the preprocessing
            pipeline.  It includes columns from W1 / D1 / H4 / H1 as well
            as the base M15 OHLCV.  All timestamps are UTC-aware.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing **only** the newly generated feature
            columns — no copies of input columns.  The returned DataFrame
            **must** share the exact same index as ``df``.

        Raises
        ------
        NotImplementedError
            If the subclass fails to implement this method.
        """
        ...

    @abstractmethod
    def metadata(self) -> FeatureMetadata:
        """Return the feature descriptor.

        Returns
        -------
        FeatureMetadata
            Fully populated metadata object.  ``name`` and ``category``
            must match the class attributes.
        """
        ...

    # ── Built-in output validation ──────────────────────────────────────────
    def validate_output(
        self,
        input_df:  pd.DataFrame,
        output_df: pd.DataFrame,
    ) -> None:
        """Lightweight post-generate sanity checks.

        Raises
        ------
        ValueError
            If the output DataFrame violates the contract.
        """
        if len(output_df) != len(input_df):
            raise ValueError(
                f"[{self.name}] generate() returned {len(output_df)} rows "
                f"but input has {len(input_df)} rows."
            )

        if not output_df.index.equals(input_df.index):
            raise ValueError(
                f"[{self.name}] generate() returned a DataFrame whose index "
                "does not match the input index."
            )

        dup_cols = output_df.columns[output_df.columns.duplicated()].tolist()
        if dup_cols:
            raise ValueError(
                f"[{self.name}] generate() returned duplicate column names: {dup_cols}"
            )

        # Warn (not raise) on column names that collide with input columns.
        overlap = set(output_df.columns) & set(input_df.columns)
        if overlap:
            logger.warning(
                "[%s] Output columns shadow input columns: %s. "
                "Use a unique prefix to avoid collisions.",
                self.name, sorted(overlap),
            )

    # ── Dunder helpers ──────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"category={self.category!r})"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BaseFeature) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
