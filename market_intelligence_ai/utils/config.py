"""MIA runtime configuration — all values overridable via environment variables."""
from __future__ import annotations

import os


class MIAConfig:
    # ── Provider ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Single analysis model — rich context does not require a fast model
    MIA_ANALYSIS_MODEL: str = os.getenv("MIA_ANALYSIS_MODEL", "llama-3.3-70b-versatile")

    MIA_TEMPERATURE: float = float(os.getenv("MIA_TEMPERATURE", "0.1"))
    MIA_MAX_TOKENS:  int   = int(os.getenv("MIA_MAX_TOKENS",    "600"))

    # ── Retry / Circuit Breaker ───────────────────────────────────────────────
    MIA_MAX_RETRIES: int             = int(os.getenv("MIA_MAX_RETRIES",                "2"))
    MIA_CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("MIA_CIRCUIT_BREAKER_THRESHOLD", "5"))
    MIA_CIRCUIT_RESET_SECONDS: int   = int(os.getenv("MIA_CIRCUIT_RESET_SECONDS",       "60"))

    # ── Cache TTLs ────────────────────────────────────────────────────────────
    MIA_EVENT_CACHE_TTL_S:    int = int(os.getenv("MIA_EVENT_CACHE_TTL_S",    "1800"))  # 30m
    MIA_HEADLINE_CACHE_TTL_S: int = int(os.getenv("MIA_HEADLINE_CACHE_TTL_S", "3600"))  # 1h
    MIA_COMBINED_CACHE_TTL_S: int = int(os.getenv("MIA_COMBINED_CACHE_TTL_S", "600"))   # 10m

    # ── Scheduler ─────────────────────────────────────────────────────────────
    MIA_CYCLE_SECONDS: int = int(os.getenv("MIA_CYCLE_SECONDS", "300"))  # 5min

    # ── Versioning ────────────────────────────────────────────────────────────
    ANALYSIS_SCHEMA_VERSION: str = "market_intelligence_ai_v1"
    CONTEXT_SCHEMA_VERSION:  str = "context_v1"
    PROVIDER_VERSION:        str = "groq_v1"

    # ── Context Builder ───────────────────────────────────────────────────────
    # Maximum number of active/upcoming events to include in context
    MIA_CONTEXT_MAX_ACTIVE_EVENTS:   int = int(os.getenv("MIA_CONTEXT_MAX_ACTIVE_EVENTS",   "5"))
    MIA_CONTEXT_MAX_UPCOMING_EVENTS: int = int(os.getenv("MIA_CONTEXT_MAX_UPCOMING_EVENTS", "3"))
    # Market session labels by UTC hour bracket
    MIA_ASIA_START_UTC:   int = int(os.getenv("MIA_ASIA_START_UTC",   "0"))
    MIA_LONDON_START_UTC: int = int(os.getenv("MIA_LONDON_START_UTC", "8"))
    MIA_NEW_YORK_START_UTC: int = int(os.getenv("MIA_NEW_YORK_START_UTC", "13"))

    @property
    def groq_configured(self) -> bool:
        return bool(self.GROQ_API_KEY)


mia_config = MIAConfig()
