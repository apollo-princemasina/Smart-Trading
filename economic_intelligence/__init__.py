"""
Economic Intelligence Engine (EIE) — Phase 3 of the MFIP Market Intelligence Layer.

Converts raw MFIPEvent objects into structured economic intelligence through
fully deterministic, explainable, and reproducible logic. No LLMs, no AI.

Entry point: economic_intelligence.engine.EconomicIntelligenceEngine
"""

__all__ = ["EconomicIntelligenceEngine"]


def __getattr__(name: str):
    if name == "EconomicIntelligenceEngine":
        from economic_intelligence.engine import EconomicIntelligenceEngine
        return EconomicIntelligenceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
