"""MIA canonical enums — used in MarketIntelligenceOutput and ContextPayload."""
from enum import Enum


class MarketBias(str, Enum):
    """Directional market bias produced by the Market Intelligence Agent."""
    BULLISH   = "BULLISH"
    BEARISH   = "BEARISH"
    NEUTRAL   = "NEUTRAL"
    UNCERTAIN = "UNCERTAIN"


class Importance(str, Enum):
    """Importance tier for the event or headline being analysed."""
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class TimeHorizon(str, Enum):
    """Expected duration of market impact."""
    IMMEDIATE   = "IMMEDIATE"    # < 1 hour
    SHORT_TERM  = "SHORT_TERM"   # 1–24 hours
    MEDIUM_TERM = "MEDIUM_TERM"  # 1–7 days
    LONG_TERM   = "LONG_TERM"    # > 7 days


class RiskLevel(str, Enum):
    """
    Risk level assigned by the Risk Manager reasoning perspective.
    Reflects whether the event increases or decreases trading risk.
    """
    LOW      = "LOW"       # Normal conditions, minimal execution risk
    MEDIUM   = "MEDIUM"    # Elevated uncertainty or volatility expected
    HIGH     = "HIGH"      # Significant contradictions or high-impact surprise
    CRITICAL = "CRITICAL"  # Extreme uncertainty, contradictions, or cluster risk


class AnalysisType(str, Enum):
    """What triggered this analysis request."""
    EVENT    = "event"
    HEADLINE = "headline"
    COMBINED = "combined"
    GENERAL  = "general"   # periodic general market narrative, no specific trigger
