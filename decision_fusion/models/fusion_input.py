"""FusionInput — the canonical input bundle consumed by the Decision Fusion Engine.

All upstream systems deposit their intelligence here before the DFE processes it.
Reserved fields are present to ensure the architecture remains extensible without
requiring structural changes to downstream consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class FusionInput:
    """
    The complete intelligence bundle passed to the Decision Fusion Engine.

    Every field is Optional so the DFE can operate gracefully even when
    one or more upstream subsystems are unavailable or have not yet produced output.
    """

    # ── Machine Learning Pipeline ─────────────────────────────────────────────
    # Raw dict returned by InferenceEngine.latest_result() (or None if no signal)
    ml_prediction: Optional[dict] = None

    # ── Economic Intelligence Engine ──────────────────────────────────────────
    # List of EconomicIntelligenceReport dataclasses from the EIE cache
    eie_reports: list = field(default_factory=list)

    # ── Market Intelligence AI ────────────────────────────────────────────────
    # MarketIntelligenceOutput Pydantic model (or None if AI unavailable)
    mia_output: Optional[Any] = None

    # ── Live Market State (Rolling Buffer) ────────────────────────────────────
    latest_close:  Optional[float] = None   # Latest M15 close price
    buffer_ready:  bool = False              # True when M15/H1/H4 buffers are populated
    buffer_status: dict = field(default_factory=dict)  # {tf: ready_bool}

    # ── Temporal ─────────────────────────────────────────────────────────────
    current_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Reserved for Future Subsystems ────────────────────────────────────────
    # These fields are intentionally Optional[dict] rather than typed models
    # to decouple the DFE from subsystems that do not yet exist.
    execution_context:  Optional[dict] = None   # Execution Context Engine (future)
    cross_asset_intel:  Optional[dict] = None   # Cross-Asset Intelligence (future)
    cot_intel:          Optional[dict] = None   # COT Intelligence (future)
    macro_intel:        Optional[dict] = None   # Macro Intelligence (future)

    # ── ML convenience properties ─────────────────────────────────────────────

    @property
    def ml_direction(self) -> Optional[str]:
        if self.ml_prediction is None:
            return None
        return self.ml_prediction.get("direction")

    @property
    def ml_confidence(self) -> Optional[float]:
        if self.ml_prediction is None:
            return None
        return self.ml_prediction.get("confidence")

    @property
    def ml_regime(self) -> Optional[str]:
        if self.ml_prediction is None:
            return None
        return self.ml_prediction.get("regime")

    @property
    def ml_session(self) -> Optional[str]:
        if self.ml_prediction is None:
            return None
        return self.ml_prediction.get("session")

    # ── EIE convenience properties ────────────────────────────────────────────

    @property
    def eie_execution_risk(self) -> float:
        """Highest execution risk from any active EIE report (0–100)."""
        if not self.eie_reports:
            return 0.0
        risks = [getattr(r, "execution_risk", 0.0) for r in self.eie_reports]
        return max(risks) if risks else 0.0

    @property
    def eie_execution_readiness(self) -> float:
        """Execution readiness from the most influential EIE report (0–100)."""
        if not self.eie_reports:
            return 100.0
        active = [
            r for r in self.eie_reports
            if getattr(r, "remaining_influence", 0.0) > 20.0
        ]
        if not active:
            return 100.0
        # Return from highest-impact active report
        top = max(active, key=lambda r: getattr(r, "impact_score", 0.0))
        return getattr(top, "execution_readiness", 100.0)

    # ── MIA convenience properties ────────────────────────────────────────────

    @property
    def mia_bias(self) -> Optional[str]:
        if self.mia_output is None:
            return None
        bias = getattr(self.mia_output, "market_bias", None)
        return bias.value if hasattr(bias, "value") else str(bias) if bias else None

    @property
    def mia_risk_level(self) -> Optional[str]:
        if self.mia_output is None:
            return None
        rl = getattr(self.mia_output, "risk_level", None)
        return rl.value if hasattr(rl, "value") else str(rl) if rl else None

    @property
    def mia_confidence(self) -> Optional[float]:
        if self.mia_output is None:
            return None
        return getattr(self.mia_output, "confidence", None)
