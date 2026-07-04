"""market_structure — ICT market structure feature generators.

Implemented
-----------
MarketStructureEngine (market_structure)
    Pivot detection (major / minor / internal), swing classification
    (HH / LH / HL / LL), trend bias, and structural reference levels.
    Single source of truth for all ICT structure features.

BosChochEngine (bos_choch)
    Break of Structure (BOS) and Change of Character (CHoCH) at both
    internal (minor pivot) and swing (major pivot) structural tiers.
    Depends on: market_structure.

OrderBlockEngine (order_blocks)
    Identifies the last opposing candle before each BOS/CHoCH and tracks
    the resulting Order Block zone until mitigated.
    Depends on: bos_choch.

FairValueGapEngine (fair_value_gaps)
    Detects three-candle price-gap patterns and tracks the most recent
    unmitigated gap zone.  Self-contained (no structural dependencies).

PremiumDiscountEngine (premium_discount)
    Classifies price as Premium / Equilibrium / Discount relative to the
    last confirmed major swing range.
    Depends on: market_structure.

Planned (Phase 5+)
------------------
- Market Structure Shift (MSS) — major trend change
- Volume-confirmed Order Blocks

Each generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

# Placeholder (smoke-test fixture)
from ._placeholder            import MarketStructurePlaceholder

# Real engines — each triggers @FeatureRegistry.register on import
from .market_structure_engine import MarketStructureEngine
from .bos_choch               import BosChochEngine
from .order_blocks            import OrderBlockEngine
from .fair_value_gaps         import FairValueGapEngine
from .premium_discount        import PremiumDiscountEngine

__all__ = [
    "MarketStructurePlaceholder",
    "MarketStructureEngine",
    "BosChochEngine",
    "OrderBlockEngine",
    "FairValueGapEngine",
    "PremiumDiscountEngine",
]
