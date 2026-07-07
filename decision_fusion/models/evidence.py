"""DFE internal evidence model — normalized representation of a single intelligence source."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from decision_fusion.models.enums import EvidenceDirection, SourceType


@dataclass(frozen=True)
class EvidenceItem:
    """
    A single piece of normalized evidence from one intelligence subsystem.

    All sources are translated into this canonical format before being
    consumed by the Agreement Engine, Confidence Engine, and Rule Engine.
    Sources are treated as independent — no averaging or voting occurs here.
    """
    source:      SourceType
    direction:   EvidenceDirection
    confidence:  float          # Source's own confidence in its reading (0–1)
    reliability: float          # Configured reliability weight for this source type (0–1)
    importance:  float          # How significant this signal is for this decision (0–1)
    timestamp:   datetime       # When this evidence was produced
    label:       str            # Human-readable identifier, e.g. "ML EURUSD M15"
    raw_value:   Optional[float] = None  # Original numeric value (e.g. ML confidence 0.73)
    metadata:    dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def is_directional(self) -> bool:
        """True when this item carries a meaningful directional signal."""
        return self.direction not in (EvidenceDirection.ABSENT, EvidenceDirection.UNCERTAIN)

    @property
    def weight(self) -> float:
        """Composite weight: reliability × importance × confidence."""
        return self.reliability * self.importance * self.confidence

    @property
    def directional_weight(self) -> float:
        """
        Signed composite weight: positive for BULLISH, negative for BEARISH, zero otherwise.
        Used by the Recommendation Engine to determine dominant direction.
        """
        if self.direction == EvidenceDirection.BULLISH:
            return self.weight
        if self.direction == EvidenceDirection.BEARISH:
            return -self.weight
        return 0.0


@dataclass
class AgreementResult:
    """Output of the Agreement Engine."""
    agreement_score:      float         # 0–100: weighted agreement across source pairs
    conflict_score:       float         # 0–100: weighted conflict across source pairs
    consensus_level:      ConsensusLevel = field(default=None)  # resolved after init
    aligned_sources:      list[str] = field(default_factory=list)
    conflicting_sources:  list[str] = field(default_factory=list)
    neutral_sources:      list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.consensus_level is None:
            from decision_fusion.models.enums import ConsensusLevel
            if self.agreement_score >= 80:
                self.consensus_level = ConsensusLevel.VERY_STRONG
            elif self.agreement_score >= 60:
                self.consensus_level = ConsensusLevel.STRONG
            elif self.agreement_score >= 40:
                self.consensus_level = ConsensusLevel.MODERATE
            else:
                self.consensus_level = ConsensusLevel.WEAK


@dataclass
class RecommendationResult:
    """Intermediate output of the Recommendation Engine (before DecisionObject assembly)."""
    recommendation: "Recommendation"
    strength:       "RecommendationStrength"
    confidence:     float           # 0–100 final decision confidence
    forced_wait:    bool = False    # True when a rule forced WAIT
