"""DFE runtime configuration — all values overridable via environment variables."""
from __future__ import annotations

import os


class DFEConfig:
    # ── Schema ────────────────────────────────────────────────────────────────
    DECISION_SCHEMA_VERSION: str = "decision_fusion_v1"

    # ── Cache ─────────────────────────────────────────────────────────────────
    DFE_HISTORY_MAX_SIZE: int = int(os.getenv("DFE_HISTORY_MAX_SIZE", "100"))

    # ── Decision Expiry (seconds) ──────────────────────────────────────────────
    # How long each decision remains valid before it should be regenerated.
    DFE_EXPIRY_WAIT_S:         int = int(os.getenv("DFE_EXPIRY_WAIT_S",         "300"))   # 5m
    DFE_EXPIRY_WEAK_S:         int = int(os.getenv("DFE_EXPIRY_WEAK_S",         "900"))   # 15m
    DFE_EXPIRY_MODERATE_S:     int = int(os.getenv("DFE_EXPIRY_MODERATE_S",     "1800"))  # 30m
    DFE_EXPIRY_STRONG_S:       int = int(os.getenv("DFE_EXPIRY_STRONG_S",       "3600"))  # 60m
    DFE_EXPIRY_VERY_STRONG_S:  int = int(os.getenv("DFE_EXPIRY_VERY_STRONG_S",  "7200"))  # 120m

    # ── Scheduler ─────────────────────────────────────────────────────────────
    DFE_CYCLE_SECONDS: int = int(os.getenv("DFE_CYCLE_SECONDS", "60"))

    # ── Evidence Source Reliability ───────────────────────────────────────────
    # How reliable each source type is considered to be (0–1).
    DFE_RELIABILITY_ML:     float = float(os.getenv("DFE_RELIABILITY_ML",     "0.85"))
    DFE_RELIABILITY_EIE:    float = float(os.getenv("DFE_RELIABILITY_EIE",    "0.75"))
    DFE_RELIABILITY_MIA:    float = float(os.getenv("DFE_RELIABILITY_MIA",    "0.70"))

    # ── Evidence Thresholds ───────────────────────────────────────────────────
    DFE_EIE_MIN_REMAINING_INFLUENCE: float = float(
        os.getenv("DFE_EIE_MIN_REMAINING_INFLUENCE", "20.0")
    )

    # ── Confidence Thresholds ─────────────────────────────────────────────────
    DFE_CONFIDENCE_MIN_THRESHOLD: float = float(os.getenv("DFE_CONFIDENCE_MIN_THRESHOLD", "30.0"))
    DFE_CONFIDENCE_CONFLICT_HIGH: float = float(os.getenv("DFE_CONFIDENCE_CONFLICT_HIGH", "70.0"))
    DFE_CONFIDENCE_CONFLICT_MED:  float = float(os.getenv("DFE_CONFIDENCE_CONFLICT_MED",  "50.0"))
    DFE_CONFIDENCE_EXECRISK_HIGH: float = float(os.getenv("DFE_CONFIDENCE_EXECRISK_HIGH", "80.0"))
    DFE_CONFIDENCE_EXECRISK_MED:  float = float(os.getenv("DFE_CONFIDENCE_EXECRISK_MED",  "50.0"))

    # ── Confidence Penalties ──────────────────────────────────────────────────
    DFE_PENALTY_HIGH_CONFLICT:   float = float(os.getenv("DFE_PENALTY_HIGH_CONFLICT",   "20.0"))
    DFE_PENALTY_MED_CONFLICT:    float = float(os.getenv("DFE_PENALTY_MED_CONFLICT",    "10.0"))
    DFE_PENALTY_HIGH_EXECRISK:   float = float(os.getenv("DFE_PENALTY_HIGH_EXECRISK",   "15.0"))
    DFE_PENALTY_MED_EXECRISK:    float = float(os.getenv("DFE_PENALTY_MED_EXECRISK",    "8.0"))
    DFE_PENALTY_AI_CRITICAL_RISK: float = float(os.getenv("DFE_PENALTY_AI_CRITICAL_RISK", "15.0"))

    # ── Confidence Bonuses ────────────────────────────────────────────────────
    DFE_BONUS_EIE_ALIGNMENT:    float = float(os.getenv("DFE_BONUS_EIE_ALIGNMENT",    "10.0"))
    DFE_BONUS_MIA_ALIGNMENT:    float = float(os.getenv("DFE_BONUS_MIA_ALIGNMENT",    "8.0"))
    DFE_BONUS_TRIPLE_CONFIRM:   float = float(os.getenv("DFE_BONUS_TRIPLE_CONFIRM",   "10.0"))
    DFE_FALLBACK_BASE_CONFIDENCE: float = float(os.getenv("DFE_FALLBACK_BASE_CONFIDENCE", "50.0"))

    # ── Rule Thresholds ───────────────────────────────────────────────────────
    DFE_RULE_CONFLICT_FORCE_WAIT:    float = float(os.getenv("DFE_RULE_CONFLICT_FORCE_WAIT",    "70.0"))
    DFE_RULE_CONFLICT_FORCE_WAIT_CONF: float = float(os.getenv("DFE_RULE_CONFLICT_FORCE_WAIT_CONF", "40.0"))
    DFE_RULE_CONFLICT_REDUCE:        float = float(os.getenv("DFE_RULE_CONFLICT_REDUCE",        "50.0"))
    DFE_RULE_EXECRISK_CRITICAL:      float = float(os.getenv("DFE_RULE_EXECRISK_CRITICAL",      "80.0"))


dfe_config = DFEConfig()
