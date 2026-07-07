"""
Business Rules — all deterministic DFE rules in one place.

Rules are never scattered throughout the codebase. Every rule lives here,
is identified by a unique ID, has a clear priority, and produces a documented
action when its condition is met.

Priority: lower number = higher priority, evaluated first.
Evaluation stops when a FORCE_WAIT rule is triggered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from decision_fusion.models.enums import ConsensusLevel, RecommendationStrength
from decision_fusion.utils.config import dfe_config


class RuleAction(str, Enum):
    """Actions a rule can take when its condition is met."""
    FORCE_WAIT        = "FORCE_WAIT"         # Override recommendation to WAIT
    REDUCE_CONFIDENCE = "REDUCE_CONFIDENCE"  # Subtract from confidence
    BOOST_CONFIDENCE  = "BOOST_CONFIDENCE"   # Add to confidence
    CAP_STRENGTH      = "CAP_STRENGTH"       # Prevent strength from exceeding a level


@dataclass(frozen=True)
class EvaluationContext:
    """
    Snapshot of key decision metrics passed to every rule condition.

    Built by the RuleEvaluator from evidence and preliminary calculations.
    All rule conditions are pure functions of this context — no side effects.
    """
    has_ml:               bool
    has_eie:              bool
    has_mia:              bool
    buffer_ready:         bool
    total_source_count:   int    # sources with non-ABSENT direction
    active_source_count:  int    # sources with BULLISH or BEARISH direction
    conflict_score:       float  # 0–100
    agreement_score:      float  # 0–100
    preliminary_confidence: float  # 0–100, before rule adjustments
    ml_direction:         Optional[str]   # BUY | SELL | HOLD | None
    fundamental_direction: Optional[str] # BULLISH | BEARISH | NEUTRAL | UNCERTAIN | None
    ai_direction:         Optional[str]  # BULLISH | BEARISH | NEUTRAL | UNCERTAIN | None
    eie_execution_risk:   float   # 0–100
    ai_risk_level:        Optional[str]  # LOW | MEDIUM | HIGH | CRITICAL | None
    consensus_level:      ConsensusLevel


@dataclass(frozen=True)
class BusinessRule:
    """A single deterministic business rule."""
    rule_id:     str
    priority:    int          # Lower = evaluated first
    description: str
    condition:   Callable[[EvaluationContext], bool] = field(compare=False, hash=False)
    action:      RuleAction
    reason:      str          # Human-readable reason shown in DecisionObject
    adjustment:  float = 0.0  # For REDUCE/BOOST actions
    strength_cap: Optional[RecommendationStrength] = None  # For CAP_STRENGTH
    stops_evaluation: bool = False  # FORCE_WAIT rules stop further evaluation


@dataclass
class RuleEvaluation:
    """Aggregate output of all triggered rules."""
    forced_wait:       bool = False
    forced_wait_reason: str = ""
    confidence_delta:  float = 0.0   # Net adjustment (sum of all reduce/boost)
    strength_cap:      Optional[RecommendationStrength] = None
    triggered_rules:   list[str] = field(default_factory=list)
    rule_reasons:      list[str] = field(default_factory=list)


# ── Rule condition functions ──────────────────────────────────────────────────
# These are pure functions of EvaluationContext — no side effects.

def _no_evidence(ctx: EvaluationContext) -> bool:
    return ctx.total_source_count == 0

def _buffer_not_ready(ctx: EvaluationContext) -> bool:
    return not ctx.buffer_ready

def _critical_conflict_low_confidence(ctx: EvaluationContext) -> bool:
    return (
        ctx.conflict_score >= dfe_config.DFE_RULE_CONFLICT_FORCE_WAIT
        and ctx.preliminary_confidence < dfe_config.DFE_RULE_CONFLICT_FORCE_WAIT_CONF
    )

def _ml_hold(ctx: EvaluationContext) -> bool:
    return ctx.has_ml and ctx.ml_direction == "HOLD"

def _confidence_too_low(ctx: EvaluationContext) -> bool:
    return ctx.preliminary_confidence < dfe_config.DFE_CONFIDENCE_MIN_THRESHOLD

def _critical_execution_risk(ctx: EvaluationContext) -> bool:
    return ctx.eie_execution_risk >= dfe_config.DFE_RULE_EXECRISK_CRITICAL

def _high_conflict_multi_source(ctx: EvaluationContext) -> bool:
    return (
        ctx.conflict_score >= dfe_config.DFE_RULE_CONFLICT_REDUCE
        and ctx.total_source_count >= 2
    )

def _triple_confirm_buy(ctx: EvaluationContext) -> bool:
    return (
        ctx.has_ml
        and ctx.has_eie
        and ctx.has_mia
        and ctx.ml_direction == "BUY"
        and ctx.fundamental_direction == "BULLISH"
        and ctx.ai_direction == "BULLISH"
    )

def _triple_confirm_sell(ctx: EvaluationContext) -> bool:
    return (
        ctx.has_ml
        and ctx.has_eie
        and ctx.has_mia
        and ctx.ml_direction == "SELL"
        and ctx.fundamental_direction == "BEARISH"
        and ctx.ai_direction == "BEARISH"
    )

def _single_source_only(ctx: EvaluationContext) -> bool:
    return ctx.active_source_count == 1

def _ai_critical_risk(ctx: EvaluationContext) -> bool:
    return ctx.ai_risk_level == "CRITICAL"

def _ml_fundamental_direct_conflict(ctx: EvaluationContext) -> bool:
    """ML says BUY but EIE says BEARISH, or ML says SELL but EIE says BULLISH."""
    return (
        ctx.has_ml
        and ctx.has_eie
        and (
            (ctx.ml_direction == "BUY"  and ctx.fundamental_direction == "BEARISH")
            or (ctx.ml_direction == "SELL" and ctx.fundamental_direction == "BULLISH")
        )
    )

def _very_strong_consensus_directional(ctx: EvaluationContext) -> bool:
    return (
        ctx.consensus_level == ConsensusLevel.VERY_STRONG
        and ctx.active_source_count >= 2
    )


# ── Rule Registry ─────────────────────────────────────────────────────────────
# Rules are evaluated in ascending priority order.
# FORCE_WAIT rules stop further evaluation — subsequent rules are not checked.

RULES: list[BusinessRule] = [
    BusinessRule(
        rule_id          = "R001",
        priority         = 1,
        description      = "No evidence — force WAIT",
        condition        = _no_evidence,
        action           = RuleAction.FORCE_WAIT,
        reason           = "No intelligence sources are available.",
        stops_evaluation = True,
    ),
    BusinessRule(
        rule_id          = "R002",
        priority         = 2,
        description      = "Buffer not ready — force WAIT",
        condition        = _buffer_not_ready,
        action           = RuleAction.FORCE_WAIT,
        reason           = "Live market data buffers are not yet populated.",
        stops_evaluation = True,
    ),
    BusinessRule(
        rule_id          = "R003",
        priority         = 5,
        description      = "Critical conflict with low confidence — force WAIT",
        condition        = _critical_conflict_low_confidence,
        action           = RuleAction.FORCE_WAIT,
        reason           = "Strong contradictions across sources with insufficient confidence prevent a reliable recommendation.",
        stops_evaluation = True,
    ),
    BusinessRule(
        rule_id          = "R004",
        priority         = 8,
        description      = "ML model recommends HOLD — force WAIT",
        condition        = _ml_hold,
        action           = RuleAction.FORCE_WAIT,
        reason           = "ML model confidence is below the execution threshold — no directional trade recommended.",
        stops_evaluation = True,
    ),
    BusinessRule(
        rule_id          = "R005",
        priority         = 10,
        description      = "Critical EIE execution risk — reduce confidence",
        condition        = _critical_execution_risk,
        action           = RuleAction.REDUCE_CONFIDENCE,
        adjustment       = -dfe_config.DFE_PENALTY_HIGH_EXECRISK,
        reason           = "Critical execution risk from upcoming or active economic events.",
    ),
    BusinessRule(
        rule_id          = "R006",
        priority         = 15,
        description      = "Direct ML–EIE conflict — reduce confidence",
        condition        = _ml_fundamental_direct_conflict,
        action           = RuleAction.REDUCE_CONFIDENCE,
        adjustment       = -dfe_config.DFE_PENALTY_MED_CONFLICT,
        reason           = "Technical (ML) and fundamental (EIE) intelligence are in direct opposition.",
    ),
    BusinessRule(
        rule_id          = "R007",
        priority         = 20,
        description      = "High multi-source conflict — reduce confidence",
        condition        = _high_conflict_multi_source,
        action           = RuleAction.REDUCE_CONFIDENCE,
        adjustment       = -dfe_config.DFE_PENALTY_MED_CONFLICT,
        reason           = "Significant disagreement detected across multiple intelligence sources.",
    ),
    BusinessRule(
        rule_id          = "R008",
        priority         = 25,
        description      = "AI identifies critical market risk — reduce confidence",
        condition        = _ai_critical_risk,
        action           = RuleAction.REDUCE_CONFIDENCE,
        adjustment       = -dfe_config.DFE_PENALTY_AI_CRITICAL_RISK,
        reason           = "Market Intelligence AI identifies critical risk conditions (extreme uncertainty or cluster risk).",
    ),
    BusinessRule(
        rule_id          = "R009",
        priority         = 30,
        description      = "Triple confirmation BUY — boost confidence",
        condition        = _triple_confirm_buy,
        action           = RuleAction.BOOST_CONFIDENCE,
        adjustment       = +dfe_config.DFE_BONUS_TRIPLE_CONFIRM,
        reason           = "Technical, fundamental, and AI intelligence all confirm bullish bias.",
    ),
    BusinessRule(
        rule_id          = "R010",
        priority         = 31,
        description      = "Triple confirmation SELL — boost confidence",
        condition        = _triple_confirm_sell,
        action           = RuleAction.BOOST_CONFIDENCE,
        adjustment       = +dfe_config.DFE_BONUS_TRIPLE_CONFIRM,
        reason           = "Technical, fundamental, and AI intelligence all confirm bearish bias.",
    ),
    BusinessRule(
        rule_id          = "R011",
        priority         = 35,
        description      = "Single active source — cap strength at MODERATE",
        condition        = _single_source_only,
        action           = RuleAction.CAP_STRENGTH,
        strength_cap     = RecommendationStrength.MODERATE,
        reason           = "Only one intelligence source is active — insufficient for strong conviction.",
    ),
    BusinessRule(
        rule_id          = "R012",
        priority         = 40,
        description      = "Very strong consensus across multiple sources — boost confidence",
        condition        = _very_strong_consensus_directional,
        action           = RuleAction.BOOST_CONFIDENCE,
        adjustment       = +5.0,
        reason           = "Very strong cross-source consensus detected.",
    ),
    BusinessRule(
        rule_id          = "R013",
        priority         = 100,
        description      = "Final confidence below minimum threshold — force WAIT",
        condition        = _confidence_too_low,
        action           = RuleAction.FORCE_WAIT,
        reason           = "Insufficient confidence to generate a reliable directional recommendation.",
        stops_evaluation = True,
    ),
]

# Sort once at module load — rules must be evaluated in priority order
RULES = sorted(RULES, key=lambda r: r.priority)
