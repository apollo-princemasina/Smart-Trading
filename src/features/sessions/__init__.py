"""sessions — Forex trading session feature generators.

Implemented
-----------
SessionEngine (sessions)
    Combines LuxAlgo Sessions and ICT Kill Zones into 25 ML-ready float64
    columns.  Detects Sydney, Asia/Tokyo, London, and New York sessions via
    UTC-hour comparison; tracks running H/L/VWAP/volume/delta per dominant
    session; outputs session timing, momentum, overlap, opening-range breakout,
    and ADR position.  Zero dependencies — all features derived from OHLCV.

Planned
-------
- SessionProfileEngine  — volume-at-price profile per session
- SessionDivergenceEngine — session vs. macro delta divergence

Each generator inherits from BaseFeature and is decorated with
@FeatureRegistry.register to self-register with the pipeline.
"""

from ._placeholder  import SessionsPlaceholder
from .session_engine import SessionEngine

__all__ = [
    "SessionsPlaceholder",
    "SessionEngine",
]
