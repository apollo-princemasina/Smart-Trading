"""
Explanation Builder — generates structured, human-readable explanations.

No AI/LLM calls are made here. The builder collects the strongest evidence,
identifies drivers of confidence, and surfaces risk factors — all deterministically.

These become inputs for future AI-generated narratives.

Output fields:
  primary_reasons     : The 2–3 strongest reasons for the recommendation
  supporting_evidence : All evidence items that agree with the recommendation
  conflicting_reasons : Evidence that contradicts the recommendation
  confidence_drivers  : What specifically drove confidence up or down
  risk_factors        : Warnings, execution risks, contradiction flags
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from decision_fusion.models.enums import EvidenceDirection, Recommendation, SourceType
from decision_fusion.models.evidence import AgreementResult, EvidenceItem
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.rule_engine.rules import RuleEvaluation
from decision_fusion.utils.logger import logger


@dataclass
class ExplanationResult:
    primary_reasons:     List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    conflicting_reasons: List[str] = field(default_factory=list)
    confidence_drivers:  List[str] = field(default_factory=list)
    risk_factors:        List[str] = field(default_factory=list)


class ExplanationBuilder:
    """
    Constructs a fully structured explanation for every Decision Object.

    The explanation is composed entirely from deterministic evidence — not from
    AI inference. It provides the 'why' behind every recommendation.
    """

    def build(
        self,
        evidence_items:   List[EvidenceItem],
        agreement_result: AgreementResult,
        rule_evaluation:  RuleEvaluation,
        recommendation:   Recommendation,
        confidence:       float,
        fusion_input:     FusionInput,
    ) -> ExplanationResult:
        result = ExplanationResult()

        rec_direction = self._rec_to_direction(recommendation)

        # ── Primary Reasons ───────────────────────────────────────────────────
        # Top evidence items supporting the recommendation, sorted by weight
        supporting = [
            item for item in evidence_items
            if item.direction == rec_direction and item.is_directional
        ]
        supporting.sort(key=lambda x: x.weight, reverse=True)

        for item in supporting[:3]:
            result.primary_reasons.append(
                f"{item.label} — {item.direction.value} ({item.confidence:.0%} confidence)"
            )

        # Rule-based primary reason (if forced WAIT)
        if rule_evaluation.forced_wait and rule_evaluation.forced_wait_reason:
            result.primary_reasons.insert(0, rule_evaluation.forced_wait_reason)

        if not result.primary_reasons:
            result.primary_reasons.append("Insufficient directional evidence to generate a recommendation.")

        # ── Supporting Evidence ───────────────────────────────────────────────
        for item in supporting:
            meta_notes = self._format_metadata(item)
            note = f"{item.label}"
            if meta_notes:
                note += f" — {meta_notes}"
            result.supporting_evidence.append(note)

        if agreement_result.aligned_sources:
            result.supporting_evidence.append(
                f"Aligned sources: {', '.join(agreement_result.aligned_sources)}"
            )

        # ── Conflicting Reasons ───────────────────────────────────────────────
        conflicting = [
            item for item in evidence_items
            if item.is_directional and item.direction != rec_direction
            and item.direction not in (EvidenceDirection.NEUTRAL,)
        ]
        for item in conflicting:
            result.conflicting_reasons.append(
                f"{item.label} — {item.direction.value} ({item.confidence:.0%} confidence)"
            )

        if agreement_result.conflicting_sources:
            result.conflicting_reasons.append(
                f"Opposing sources: {', '.join(agreement_result.conflicting_sources)}"
            )

        # ── Confidence Drivers ────────────────────────────────────────────────
        result.confidence_drivers.extend(
            self._build_confidence_drivers(
                evidence_items, agreement_result, rule_evaluation, confidence, fusion_input
            )
        )

        # ── Risk Factors ──────────────────────────────────────────────────────
        result.risk_factors.extend(
            self._build_risk_factors(evidence_items, fusion_input, rule_evaluation)
        )

        logger.debug(
            "Explanation built: {} primary, {} supporting, {} conflicting, {} risk",
            len(result.primary_reasons),
            len(result.supporting_evidence),
            len(result.conflicting_reasons),
            len(result.risk_factors),
        )
        return result

    # ── Confidence Drivers ────────────────────────────────────────────────────

    def _build_confidence_drivers(
        self,
        items:           List[EvidenceItem],
        agreement:       AgreementResult,
        rule_eval:       RuleEvaluation,
        confidence:      float,
        fi:              FusionInput,
    ) -> List[str]:
        drivers: List[str] = []

        # ML anchor
        ml = next((i for i in items if i.source == SourceType.TECHNICAL_ML
                   and i.direction != EvidenceDirection.ABSENT), None)
        if ml:
            drivers.append(
                f"ML model anchor: {ml.direction.value} with {ml.confidence:.0%} confidence"
            )

        # Consensus
        drivers.append(
            f"Consensus: {agreement.consensus_level.value} "
            f"(agreement {agreement.agreement_score:.0f}%, conflict {agreement.conflict_score:.0f}%)"
        )

        # Rule bonuses / penalties
        for rule_id, reason in zip(rule_eval.triggered_rules, rule_eval.rule_reasons):
            from decision_fusion.rule_engine.rules import RULES
            rule = next((r for r in RULES if r.rule_id == rule_id), None)
            if rule and hasattr(rule, "adjustment") and rule.adjustment != 0:
                sign = "+" if rule.adjustment > 0 else ""
                drivers.append(
                    f"Rule {rule_id}: {sign}{rule.adjustment:.1f} pts — {reason}"
                )

        # Execution risk
        exec_risk = fi.eie_execution_risk
        if exec_risk > 50:
            drivers.append(
                f"EIE execution risk: {exec_risk:.0f}/100 applied as confidence penalty"
            )

        drivers.append(f"Final decision confidence: {confidence:.1f}/100")
        return drivers

    # ── Risk Factors ──────────────────────────────────────────────────────────

    def _build_risk_factors(
        self,
        items:     List[EvidenceItem],
        fi:        FusionInput,
        rule_eval: RuleEvaluation,
    ) -> List[str]:
        risks: List[str] = []

        # AI execution warning
        for item in items:
            if item.source == SourceType.AI_INTELLIGENCE:
                warn = item.metadata.get("execution_warning")
                if warn:
                    risks.append(f"AI Execution Warning: {warn}")

        # EIE execution risk
        exec_risk = fi.eie_execution_risk
        if exec_risk > 70:
            risks.append(
                f"Critical economic execution risk: {exec_risk:.0f}/100 — "
                "major event cluster or upcoming high-impact release"
            )
        elif exec_risk > 50:
            risks.append(
                f"Elevated economic execution risk: {exec_risk:.0f}/100"
            )

        # AI risk level
        ai_risk = fi.mia_risk_level
        if ai_risk in ("HIGH", "CRITICAL"):
            risks.append(
                f"AI-assessed market risk level: {ai_risk}"
            )

        # Contradictions
        contradicts = getattr(fi.mia_output, "contradicts_existing_bias", False)
        if contradicts:
            risks.append(
                "MIA: current AI analysis contradicts existing market bias"
            )

        # Rule-triggered risks
        for rule_id, reason in zip(rule_eval.triggered_rules, rule_eval.rule_reasons):
            from decision_fusion.rule_engine.rules import RULES, RuleAction
            rule = next((r for r in RULES if r.rule_id == rule_id), None)
            if rule and rule.action == RuleAction.REDUCE_CONFIDENCE:
                risks.append(f"Risk: {reason}")

        return risks

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rec_to_direction(rec: Recommendation) -> EvidenceDirection:
        if rec == Recommendation.BUY:
            return EvidenceDirection.BULLISH
        if rec == Recommendation.SELL:
            return EvidenceDirection.BEARISH
        return EvidenceDirection.NEUTRAL

    @staticmethod
    def _format_metadata(item: EvidenceItem) -> str:
        meta = item.metadata
        parts = []
        if item.source == SourceType.TECHNICAL_ML:
            regime = meta.get("regime")
            session = meta.get("session")
            if regime:
                parts.append(f"regime={regime}")
            if session:
                parts.append(f"session={session}")
        elif item.source == SourceType.FUNDAMENTAL_EIE:
            count = meta.get("active_event_count")
            top   = meta.get("top_event_title", "")
            if top:
                parts.append(f"top event: {top}")
            if count:
                parts.append(f"{count} active events")
        elif item.source == SourceType.AI_INTELLIGENCE:
            summary = meta.get("market_summary", "")
            if summary:
                parts.append(summary[:100])
        return "; ".join(parts)
