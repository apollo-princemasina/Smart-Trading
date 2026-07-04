"""Market Structure Engine — the single source of truth for ICT structure.

This is the first real feature generator in the pipeline.  It orchestrates
four specialist sub-modules:

    PivotDetector   — confirmed swing highs and lows at three strength tiers
    SwingAnalyzer   — HH / LH / HL / LL classification + swing metadata
    TrendEngine     — market bias (bullish / bearish / neutral) from swings
    StructureState  — last key price levels + distance-to-structure features

Output (31 columns, all float64)
---------------------------------
From PivotDetector (6):
    pivot_high, pivot_low
    major_pivot_high, major_pivot_low
    minor_pivot_high, minor_pivot_low

From SwingAnalyzer (14):
    higher_high, lower_high, higher_low, lower_low
    swing_high_id, swing_low_id
    swing_high_price, swing_low_price
    swing_high_duration, swing_low_duration
    swing_high_range, swing_low_range
    swing_high_strength, swing_low_strength

From TrendEngine (3):
    trend, trend_duration, trend_strength

From StructureState (8):
    last_major_high, last_major_low
    last_internal_high, last_internal_low
    distance_to_last_major_high, distance_to_last_major_low
    distance_to_last_internal_high, distance_to_last_internal_low

Design constraints
------------------
* No repainting: pivot detection uses only bars [i-lookback, i+lookback].
  For bar i, every right-side bar used in confirmation is already in the
  historical record at evaluation time.
* No BOS / CHoCH / Order Blocks / FVGs — those are Phase 5 features.
* No plotting, no TradingView objects, no Pine Script references.
* Inherits BaseFeature and self-registers via @FeatureRegistry.register.
"""

from __future__ import annotations

import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry
from .pivots            import PivotConfig, PivotDetector
from .swings            import SwingAnalyzer
from .trend             import TrendEngine
from .structure         import StructureState

# ── Output column catalogue (for metadata and validation) ─────────────────────

_PIVOT_COLS: list[str] = [
    "pivot_high", "pivot_low",
    "major_pivot_high", "major_pivot_low",
    "minor_pivot_high", "minor_pivot_low",
]
_SWING_COLS: list[str] = [
    "higher_high", "lower_high", "higher_low", "lower_low",
    "swing_high_id", "swing_low_id",
    "swing_high_price", "swing_low_price",
    "swing_high_duration", "swing_low_duration",
    "swing_high_range", "swing_low_range",
    "swing_high_strength", "swing_low_strength",
]
_TREND_COLS: list[str] = [
    "trend", "trend_duration", "trend_strength",
]
_STRUCTURE_COLS: list[str] = [
    "last_major_high", "last_major_low",
    "last_internal_high", "last_internal_low",
    "distance_to_last_major_high", "distance_to_last_major_low",
    "distance_to_last_internal_high", "distance_to_last_internal_low",
]
_ALL_OUTPUT_COLS: list[str] = (
    _PIVOT_COLS + _SWING_COLS + _TREND_COLS + _STRUCTURE_COLS
)


@FeatureRegistry.register
class MarketStructureEngine(BaseFeature):
    """Institutional market structure foundation for ICT-based strategies.

    Detects and classifies the structural backbone of price action (swing
    highs, swing lows, trend bias, and structural reference levels) that all
    future ICT feature modules will depend on.

    Configuration
    -------------
    major_lookback : int
        Lookback bars for major (significant) pivot detection.  Default 15
        (~3.75 h on M15).
    minor_lookback : int
        Lookback bars for minor (standard) pivot detection.  Default 5
        (~1.25 h on M15).
    internal_lookback : int
        Lookback bars for internal (micro) pivot detection.  Default 3
        (~45 min on M15).

    The engine can be instantiated with default lookbacks by calling
    ``MarketStructureEngine()`` (no arguments) — the pipeline always does
    this when executing the feature.
    """

    # ── BaseFeature contract ───────────────────────────────────────────────────
    name:             str       = "market_structure"
    category:         str       = "market_structure"
    dependencies:     list[str] = []
    required_columns: list[str] = ["high", "low", "close"]

    # ── Default configuration ──────────────────────────────────────────────────
    _DEFAULT_MAJOR_LB:    int = 15
    _DEFAULT_MINOR_LB:    int =  5
    _DEFAULT_INTERNAL_LB: int =  3

    def __init__(
        self,
        major_lookback:    int = _DEFAULT_MAJOR_LB,
        minor_lookback:    int = _DEFAULT_MINOR_LB,
        internal_lookback: int = _DEFAULT_INTERNAL_LB,
    ) -> None:
        config = PivotConfig(
            major_lookback    = major_lookback,
            minor_lookback    = minor_lookback,
            internal_lookback = internal_lookback,
        )
        self._detector  = PivotDetector(config)
        self._swings    = SwingAnalyzer()
        self._trend     = TrendEngine()
        self._structure = StructureState()

    # ── Core computation ───────────────────────────────────────────────────────

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full market structure pipeline and return all output columns.

        Pipeline
        --------
        1. PivotDetector  → confirmed pivot highs and lows (3 strength tiers)
        2. SwingAnalyzer  → HH/LH/HL/LL + swing metadata (uses minor pivots)
        3. TrendEngine    → trend direction / duration / strength
        4. StructureState → last key levels + % distances to close

        Parameters
        ----------
        df:
            The merged M15 OHLCV DataFrame from the preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
            31 float64 columns, same index as ``df``.  Internal-use pivot
            columns (``_internal_pivot_*``) are NOT included in the output.
        """
        # ── Step 1: detect pivots ──────────────────────────────────────────
        pivot_df = self._detector.detect(df)

        # ── Step 2: classify swings (operate on MINOR pivot tier) ─────────
        swing_df = self._swings.analyze(
            df,
            pivot_highs = pivot_df["minor_pivot_high"].astype(bool),
            pivot_lows  = pivot_df["minor_pivot_low"].astype(bool),
        )

        # ── Step 3: derive trend from swing sequence ───────────────────────
        trend_df = self._trend.compute(df, swing_df)

        # ── Step 4: track structural reference levels ──────────────────────
        structure_df = self._structure.compute(df, pivot_df)

        # ── Combine, strip internal columns, return ────────────────────────
        public_pivot_df = pivot_df.drop(
            columns=["_internal_pivot_high", "_internal_pivot_low"],
        )
        return pd.concat([public_pivot_df, swing_df, trend_df, structure_df], axis=1)

    # ── Metadata ───────────────────────────────────────────────────────────────

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Institutional market structure engine.  Detects confirmed "
                "pivot highs and lows at major / minor / internal strength "
                "tiers, classifies each pivot as HH / LH / HL / LL, derives "
                "market trend bias (bullish / bearish / neutral), and tracks "
                "the last structural reference levels with distance features. "
                "Single source of truth for all future ICT feature modules."
            ),
            dependencies     = [],
            required_columns = self.required_columns,
            output_columns   = _ALL_OUTPUT_COLS,
            version          = "1.0.0",
            author           = "Smart Trading Team",
            complexity       = "high",
            tags             = [
                "ICT", "smart_money", "market_structure",
                "pivot", "swing", "trend", "structure",
                "HH", "HL", "LH", "LL",
            ],
        )
