"""
Market Intelligence AI Layer (MIA) — Phase 4 of the MFIP intelligence pipeline.

Architecture: Context-Driven Autonomous Market Intelligence Agent.

Instead of multiple specialized analysts with hardcoded prompts, the MIA uses:
  - ContextBuilder: dynamically assembles all available market information
  - Single system prompt: defines the AI's permanent institutional role
  - MarketIntelligenceAgent: reasons autonomously from the assembled context
  - Single output schema: consistent MarketIntelligenceOutput regardless of input type

IMPORTANT CONSTRAINTS (enforced at every level):
  - The AI is NOT responsible for trading decisions
  - The AI NEVER produces BUY, SELL, HOLD, LONG, SHORT, ENTER, or EXIT recommendations
  - All outputs are structured JSON — no free-form text consumed by backend services
  - AI analysis ENRICHES deterministic EIE output — never replaces it

Pipeline position:
  Forex Factory Connector
    → Economic Intelligence Engine
      → Market Intelligence AI Layer       ← this module
        → Execution Context Engine (future)
          → Decision Fusion Engine (future)
"""

__all__ = ["MarketIntelligenceAIEngine"]


def __getattr__(name: str):
    if name == "MarketIntelligenceAIEngine":
        from market_intelligence_ai.engine import MarketIntelligenceAIEngine
        return MarketIntelligenceAIEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
