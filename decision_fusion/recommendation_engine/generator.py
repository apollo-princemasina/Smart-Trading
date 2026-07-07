"""
Recommendation Generator — produces the final BUY / SELL / WAIT recommendation.

Inputs:
  - Evidence items (normalized from all sources)
  - Agreement result (consensus level, aligned/conflicting sources)
  - Preliminary confidence (from Confidence Engine)
  - Rule evaluation (adjustments, forced WAIT, strength cap)

Output:
  - RecommendationResult (recommendation, strength, final confidence)

The generator also computes:
  - technical_alignment:   -1.0 to +1.0 (from ML pipeline)
  - fundamental_alignment: -1.0 to +1.0 (from EIE)
  - market_bias:           Canonical market bias label
  - expires_at:            When this recommendation should be considered stale
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from decision_fusion.models.enums import (
    ConsensusLevel,
    EvidenceDirection,
    MarketBiasEnum,
    Recommendation,
    RecommendationStrength,
    SourceType,
)
from decision_fusion.models.evidence import AgreementResult, EvidenceItem, RecommendationResult
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.rule_engine.rules import RuleEvaluation
from decision_fusion.utils.config import dfe_config
from decision_fusion.utils.logger import logger


class RecommendationGenerator:
    """
    Translates normalized evidence and rule outputs into a final, actionable recommendation.
    """

    def generate(
        self,
        evidence_items:    List[EvidenceItem],
        agreement_result:  AgreementResult,
        preliminary_conf:  float,
        rule_evaluation:   RuleEvaluation,
    ) -> RecommendationResult:
        """
        Produce the final RecommendationResult.

        If a rule forced WAIT, the recommendation is WAIT with the original confidence.
        Otherwise, dominant direction across evidence determines the recommendation.
        """
        # Apply rule confidence delta to get final confidence
        final_conf = max(0.0, min(100.0, preliminary_conf + rule_evaluation.confidence_delta))

        if rule_evaluation.forced_wait:
            return RecommendationResult(
                recommendation = Recommendation.WAIT,
                strength       = RecommendationStrength.WEAK,
                confidence     = final_conf,
                forced_wait    = True,
            )

        # Determine dominant direction from weighted evidence
        dominant = self._dominant_direction(evidence_items)

        if dominant == EvidenceDirection.BULLISH:
            rec = Recommendation.BUY
        elif dominant == EvidenceDirection.BEARISH:
            rec = Recommendation.SELL
        else:
            rec = Recommendation.WAIT

        # If WAIT, strength is not meaningful — use WEAK
        if rec == Recommendation.WAIT:
            return RecommendationResult(
                recommendation = Recommendation.WAIT,
                strength       = RecommendationStrength.WEAK,
                confidence     = final_conf,
                forced_wait    = False,
            )

        # Determine strength from final confidence + consensus
        strength = self._strength(final_conf, agreement_result.consensus_level)

        # Apply strength cap from rules
        if rule_evaluation.strength_cap is not None:
            if rule_evaluation.strength_cap < strength:
                strength = rule_evaluation.strength_cap

        logger.debug(
            "Recommendation: {} {} conf={:.1f}",
            rec.value, strength.value, final_conf,
        )
        return RecommendationResult(
            recommendation = rec,
            strength       = strength,
            confidence     = final_conf,
            forced_wait    = False,
        )

    def compute_technical_alignment(self, fi: FusionInput) -> float:
        """
        ML directional alignment: +1.0 = fully bullish, -1.0 = fully bearish, 0.0 = neutral.
        """
        direction = fi.ml_direction
        confidence = fi.ml_confidence or 0.0
        if direction == "BUY":
            return round(confidence, 3)
        if direction == "SELL":
            return round(-confidence, 3)
        return 0.0

    def compute_fundamental_alignment(self, fi: FusionInput) -> float:
        """
        EIE directional alignment: weighted average across all active reports.
        """
        if not fi.eie_reports:
            return 0.0

        total_weight = 0.0
        signed_weight = 0.0

        for r in fi.eie_reports:
            impact   = getattr(r, "impact_score", 50.0) / 100.0
            remain   = getattr(r, "remaining_influence", 50.0) / 100.0
            dir_conf = getattr(r, "direction_confidence", 0.5)
            raw_dir  = getattr(r, "economic_direction", None)
            dir_str  = (raw_dir.value if hasattr(raw_dir, "value") else str(raw_dir)).upper()

            w = impact * remain

            if dir_str == "BULLISH":
                signed_weight += w * dir_conf
            elif dir_str == "BEARISH":
                signed_weight -= w * dir_conf
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        result = signed_weight / total_weight
        return round(max(-1.0, min(1.0, result)), 3)

    def compute_market_bias(
        self,
        fi: FusionInput,
        dominant: EvidenceDirection,
    ) -> MarketBiasEnum:
        """
        Market bias label: use MIA's market_bias if available, otherwise derive from dominant direction.
        """
        if fi.mia_output is not None and not getattr(fi.mia_output, "is_fallback", False):
            raw_bias = fi.mia_bias
            if raw_bias:
                mapping = {
                    "BULLISH":   MarketBiasEnum.BULLISH,
                    "BEARISH":   MarketBiasEnum.BEARISH,
                    "NEUTRAL":   MarketBiasEnum.NEUTRAL,
                    "UNCERTAIN": MarketBiasEnum.UNCERTAIN,
                }
                return mapping.get(raw_bias.upper(), MarketBiasEnum.UNCERTAIN)

        # Derive from dominant direction
        if dominant == EvidenceDirection.BULLISH:
            return MarketBiasEnum.BULLISH
        if dominant == EvidenceDirection.BEARISH:
            return MarketBiasEnum.BEARISH
        if dominant == EvidenceDirection.NEUTRAL:
            return MarketBiasEnum.NEUTRAL
        return MarketBiasEnum.UNCERTAIN

    def compute_expiry(
        self,
        rec: Recommendation,
        strength: RecommendationStrength,
        now: datetime,
    ) -> datetime:
        """
        Compute when this recommendation should be considered stale.
        WAIT expires quickly; strong directional signals last longer.
        """
        if rec == Recommendation.WAIT:
            ttl = dfe_config.DFE_EXPIRY_WAIT_S
        elif strength == RecommendationStrength.WEAK:
            ttl = dfe_config.DFE_EXPIRY_WEAK_S
        elif strength == RecommendationStrength.MODERATE:
            ttl = dfe_config.DFE_EXPIRY_MODERATE_S
        elif strength == RecommendationStrength.STRONG:
            ttl = dfe_config.DFE_EXPIRY_STRONG_S
        else:  # VERY_STRONG
            ttl = dfe_config.DFE_EXPIRY_VERY_STRONG_S
        return now + timedelta(seconds=ttl)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dominant_direction(items: List[EvidenceItem]) -> EvidenceDirection:
        weights: dict[EvidenceDirection, float] = {
            EvidenceDirection.BULLISH: 0.0,
            EvidenceDirection.BEARISH: 0.0,
            EvidenceDirection.NEUTRAL: 0.0,
        }
        for item in items:
            if item.direction in weights:
                weights[item.direction] += item.reliability * item.importance * item.confidence

        bull = weights[EvidenceDirection.BULLISH]
        bear = weights[EvidenceDirection.BEARISH]
        neut = weights[EvidenceDirection.NEUTRAL]

        if bull > bear and bull > neut and bull > 0:
            return EvidenceDirection.BULLISH
        if bear > bull and bear > neut and bear > 0:
            return EvidenceDirection.BEARISH
        return EvidenceDirection.NEUTRAL

    @staticmethod
    def _strength(confidence: float, consensus: ConsensusLevel) -> RecommendationStrength:
        if confidence >= 75:
            base = RecommendationStrength.VERY_STRONG
        elif confidence >= 60:
            base = RecommendationStrength.STRONG
        elif confidence >= 45:
            base = RecommendationStrength.MODERATE
        else:
            base = RecommendationStrength.WEAK

        # Consensus caps strength — high confidence with weak consensus stays MODERATE
        consensus_cap = {
            ConsensusLevel.VERY_STRONG: RecommendationStrength.VERY_STRONG,
            ConsensusLevel.STRONG:      RecommendationStrength.STRONG,
            ConsensusLevel.MODERATE:    RecommendationStrength.MODERATE,
            ConsensusLevel.WEAK:        RecommendationStrength.MODERATE,
        }
        cap = consensus_cap.get(consensus, RecommendationStrength.MODERATE)
        return cap if cap < base else base
