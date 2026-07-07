"""DFE canonical enums — used throughout the Decision Fusion Engine."""
from enum import Enum


class Recommendation(str, Enum):
    """The final tradeable recommendation produced by the DFE."""
    BUY  = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


class RecommendationStrength(str, Enum):
    """How strongly the DFE holds its recommendation."""
    WEAK        = "WEAK"        # < 45 confidence or single source
    MODERATE    = "MODERATE"    # 45–60 confidence or limited agreement
    STRONG      = "STRONG"      # 60–75 confidence + multi-source agreement
    VERY_STRONG = "VERY_STRONG" # > 75 confidence + strong consensus

    def __lt__(self, other: "RecommendationStrength") -> bool:
        _order = [
            RecommendationStrength.WEAK,
            RecommendationStrength.MODERATE,
            RecommendationStrength.STRONG,
            RecommendationStrength.VERY_STRONG,
        ]
        return _order.index(self) < _order.index(other)

    def __le__(self, other: "RecommendationStrength") -> bool:
        return self == other or self < other


class ConsensusLevel(str, Enum):
    """Degree of agreement across all active intelligence sources."""
    WEAK        = "WEAK"        # < 40 agreement score
    MODERATE    = "MODERATE"    # 40–60 agreement score
    STRONG      = "STRONG"      # 60–80 agreement score
    VERY_STRONG = "VERY_STRONG" # > 80 agreement score


class SourceType(str, Enum):
    """Identifies which intelligence subsystem produced an evidence item."""
    TECHNICAL_ML    = "TECHNICAL_ML"    # ML prediction pipeline
    FUNDAMENTAL_EIE = "FUNDAMENTAL_EIE" # Economic Intelligence Engine
    AI_INTELLIGENCE = "AI_INTELLIGENCE" # Market Intelligence AI (Groq)
    MARKET_STATE    = "MARKET_STATE"    # Live rolling buffer state

    # Reserved for future subsystems
    EXECUTION       = "EXECUTION"       # Execution Context Engine (future)
    CROSS_ASSET     = "CROSS_ASSET"     # Cross-Asset Intelligence (future)
    COT             = "COT"             # COT Intelligence (future)
    MACRO           = "MACRO"           # Macro Intelligence (future)


class EvidenceDirection(str, Enum):
    """Normalized directional signal — common language across all source types."""
    BULLISH   = "BULLISH"   # Source signals upward price pressure
    BEARISH   = "BEARISH"   # Source signals downward price pressure
    NEUTRAL   = "NEUTRAL"   # Source signals no directional pressure
    UNCERTAIN = "UNCERTAIN" # Source is unable to determine direction
    ABSENT    = "ABSENT"    # Source provided no evidence (not available)


class MarketBiasEnum(str, Enum):
    """Overall market bias label — used in DecisionObject for downstream consumers."""
    BULLISH   = "BULLISH"
    BEARISH   = "BEARISH"
    NEUTRAL   = "NEUTRAL"
    UNCERTAIN = "UNCERTAIN"
