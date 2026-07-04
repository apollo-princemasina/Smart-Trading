"""liquidity — buy-side and sell-side liquidity feature generators.

Implemented
-----------
EqualHighsLowsEngine (equal_highs_lows)
    Detects Equal Highs (EQH) and Equal Lows (EQL) — consecutive confirmed
    pivot highs/lows within a threshold of each other.  These represent
    liquidity pools (accumulated stop orders) that price tends to sweep
    before a directional move.
    Depends on: market_structure.

LiquiditySweepEngine (liquidity_sweeps)
    Tracks buy-side and sell-side liquidity levels (pivot highs/lows, EQH/EQL)
    and detects when price sweeps (takes out) those levels.  Outputs composite
    sweep strength, nearest-pool distances, and ML-ready sweep flags.
    Depends on: market_structure, bos_choch, equal_highs_lows.

LiquidityMagnetEngine (liquidity_magnet)
    Ranks every resting liquidity pool 0-100 by magnetic pull strength and
    identifies the single NEXT TARGET.  Score = proximity (55 pts) + momentum
    (25 pts) + age (10 pts) + touches (10 pts).  Pools ≤ 5 % from close and
    scoring ≥ 35 qualify as target; outputs 20 ML-ready numerical features.
    Depends on: market_structure, bos_choch, equal_highs_lows, liquidity_sweeps.

Planned
-------
- LiquidityPoolEngine  — map buy-side and sell-side liquidity pool zones
- BuySideLiquidityEngine  — institutional buy-side targets above swing highs
- SellSideLiquidityEngine — institutional sell-side targets below swing lows

Each generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder       import LiquidityPlaceholder
from .equal_highs_lows   import EqualHighsLowsEngine
from .liquidity_sweeps   import LiquiditySweepEngine
from .liquidity_magnet   import LiquidityMagnetEngine

__all__ = [
    "LiquidityPlaceholder",
    "EqualHighsLowsEngine",
    "LiquiditySweepEngine",
    "LiquidityMagnetEngine",
]
