"""
Rule Evaluator — applies all business rules to an EvaluationContext.

Rules are stateless and deterministic. The evaluator:
  1. Builds an EvaluationContext from evidence and preliminary calculations.
  2. Evaluates rules in ascending priority order.
  3. Applies each triggered rule's action (adjust confidence, cap strength, force WAIT).
  4. Stops on the first FORCE_WAIT rule.
  5. Returns a RuleEvaluation summarising all triggered rules.
"""
from __future__ import annotations

from typing import List, Optional

from decision_fusion.models.enums import ConsensusLevel, EvidenceDirection, SourceType
from decision_fusion.models.evidence import AgreementResult, EvidenceItem
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.rule_engine.rules import (
    RULES,
    EvaluationContext,
    RuleAction,
    RuleEvaluation,
)
from decision_fusion.utils.logger import logger


class RuleEvaluator:
    """
    Applies all registered business rules to an EvaluationContext.

    The evaluator is stateless — calling evaluate() with the same inputs
    always produces the same output.
    """

    def build_context(
        self,
        evidence_items:    List[EvidenceItem],
        agreement_result:  AgreementResult,
        preliminary_conf:  float,
        fusion_input:      FusionInput,
    ) -> EvaluationContext:
        """Build an EvaluationContext from evidence and intermediate calculations."""
        by_source = {item.source: item for item in evidence_items}

        # Determine which sources are active (non-absent, non-uncertain)
        non_absent = [
            i for i in evidence_items
            if i.direction != EvidenceDirection.ABSENT
        ]
        directional = [
            i for i in evidence_items
            if i.direction in (EvidenceDirection.BULLISH, EvidenceDirection.BEARISH)
        ]

        ml_item  = by_source.get(SourceType.TECHNICAL_ML)
        eie_item = by_source.get(SourceType.FUNDAMENTAL_EIE)
        mia_item = by_source.get(SourceType.AI_INTELLIGENCE)

        def _direction_str(item: Optional[EvidenceItem]) -> Optional[str]:
            if item is None or item.direction == EvidenceDirection.ABSENT:
                return None
            return item.direction.value

        return EvaluationContext(
            has_ml               = ml_item is not None and ml_item.direction != EvidenceDirection.ABSENT,
            has_eie              = eie_item is not None and eie_item.direction != EvidenceDirection.ABSENT,
            has_mia              = mia_item is not None and mia_item.direction != EvidenceDirection.ABSENT,
            buffer_ready         = fusion_input.buffer_ready,
            total_source_count   = len(non_absent),
            active_source_count  = len(directional),
            conflict_score       = agreement_result.conflict_score,
            agreement_score      = agreement_result.agreement_score,
            preliminary_confidence = preliminary_conf,
            ml_direction         = fusion_input.ml_direction,
            fundamental_direction = _direction_str(eie_item),
            ai_direction         = _direction_str(mia_item),
            eie_execution_risk   = fusion_input.eie_execution_risk,
            ai_risk_level        = fusion_input.mia_risk_level,
            consensus_level      = agreement_result.consensus_level,
        )

    def evaluate(self, context: EvaluationContext) -> RuleEvaluation:
        """
        Evaluate all rules against the context.

        Returns a RuleEvaluation summarising every triggered rule.
        Evaluation stops immediately when a FORCE_WAIT rule triggers.
        """
        result = RuleEvaluation()

        for rule in RULES:
            try:
                if not rule.condition(context):
                    continue
            except Exception as exc:
                logger.warning("Rule {} condition raised: {}", rule.rule_id, exc)
                continue

            result.triggered_rules.append(rule.rule_id)
            result.rule_reasons.append(rule.reason)
            logger.debug("Rule {} triggered: {}", rule.rule_id, rule.description)

            if rule.action == RuleAction.FORCE_WAIT:
                result.forced_wait        = True
                result.forced_wait_reason = rule.reason
                if rule.stops_evaluation:
                    break  # No further rules evaluated after a stopping FORCE_WAIT

            elif rule.action == RuleAction.REDUCE_CONFIDENCE:
                result.confidence_delta += rule.adjustment  # adjustment is negative

            elif rule.action == RuleAction.BOOST_CONFIDENCE:
                result.confidence_delta += rule.adjustment  # adjustment is positive

            elif rule.action == RuleAction.CAP_STRENGTH:
                if rule.strength_cap is not None:
                    if (
                        result.strength_cap is None
                        or rule.strength_cap < result.strength_cap
                    ):
                        result.strength_cap = rule.strength_cap

        logger.debug(
            "Rule evaluation complete: {} rules triggered, forced_wait={}, delta={:+.1f}",
            len(result.triggered_rules),
            result.forced_wait,
            result.confidence_delta,
        )
        return result
