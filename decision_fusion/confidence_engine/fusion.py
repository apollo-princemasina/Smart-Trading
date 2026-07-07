"""
Confidence Fusion — produces a single Decision Confidence score (0–100).

Methodology:
  1. BASE: ML confidence is the primary anchor (most reliable technical signal).
     If ML is unavailable, MIA confidence is used as the base.
     If neither is available, a conservative default is applied.

  2. ALIGNMENT BONUS: When EIE or MIA align with the base direction,
     moderate bonuses are added to reflect multi-source confirmation.

  3. AGREEMENT MULTIPLIER: The consensus level scales the base up or down
     depending on how consistently all sources agree.

  4. CONFLICT PENALTY: A direct deduction for high cross-source conflict.

  5. EXECUTION RISK PENALTY: EIE execution risk reduces confidence
     because it signals market conditions that raise trading risk.

  6. AI RISK PENALTY: MIA's Risk Manager perspective (risk_level = CRITICAL)
     signals extreme market uncertainty — applied as a deduction.

  7. CLIP: Result is clamped to [0, 100].

This methodology is deterministic and reproducible — given the same inputs,
the same confidence is always produced.
"""
from __future__ import annotations

from typing import List

from decision_fusion.models.enums import ConsensusLevel, EvidenceDirection, SourceType
from decision_fusion.models.evidence import AgreementResult, EvidenceItem
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.utils.config import dfe_config
from decision_fusion.utils.logger import logger


class ConfidenceFusion:
    """
    Fuses evidence quality, agreement, and market conditions into a single
    Decision Confidence score.

    The ML model confidence is not simply averaged — it is used as a base
    and then adjusted by the quality of the surrounding evidence environment.
    """

    def compute(
        self,
        evidence_items:  List[EvidenceItem],
        agreement_result: AgreementResult,
        fusion_input:    FusionInput,
    ) -> float:
        """
        Returns a Decision Confidence score in [0.0, 100.0].

        Parameters
        ----------
        evidence_items   : normalized evidence from all sources
        agreement_result : output of the Agreement Engine
        fusion_input     : full input bundle (for execution risk, MIA risk level)
        """
        by_source = {item.source: item for item in evidence_items}

        # ── Step 1: Establish base confidence ────────────────────────────────
        base = self._base_confidence(by_source, fusion_input)

        # ── Step 2: Alignment bonuses ─────────────────────────────────────────
        dominant = self._dominant_direction(evidence_items)
        base = self._apply_alignment_bonuses(base, by_source, dominant, fusion_input)

        # ── Step 3: Agreement multiplier ──────────────────────────────────────
        base = self._apply_agreement_multiplier(base, agreement_result.consensus_level)

        # ── Step 4: Conflict penalty ───────────────────────────────────────────
        base = self._apply_conflict_penalty(base, agreement_result.conflict_score)

        # ── Step 5: Execution risk penalty ────────────────────────────────────
        base = self._apply_execution_risk_penalty(base, fusion_input.eie_execution_risk)

        # ── Step 6: AI risk level penalty ─────────────────────────────────────
        base = self._apply_ai_risk_penalty(base, fusion_input.mia_risk_level)

        # ── Step 7: Clip ───────────────────────────────────────────────────────
        result = max(0.0, min(100.0, base))
        logger.debug("Confidence fusion: {:.1f}", result)
        return round(result, 2)

    # ── Step implementations ──────────────────────────────────────────────────

    def _base_confidence(
        self,
        by_source: dict,
        fi: FusionInput,
    ) -> float:
        ml_item = by_source.get(SourceType.TECHNICAL_ML)
        if ml_item and ml_item.direction != EvidenceDirection.ABSENT and fi.ml_confidence:
            return fi.ml_confidence * 100.0

        ai_item = by_source.get(SourceType.AI_INTELLIGENCE)
        if ai_item and ai_item.direction != EvidenceDirection.ABSENT and fi.mia_confidence:
            return fi.mia_confidence * 100.0

        return dfe_config.DFE_FALLBACK_BASE_CONFIDENCE

    def _apply_alignment_bonuses(
        self,
        base: float,
        by_source: dict,
        dominant: EvidenceDirection,
        fi: FusionInput,
    ) -> float:
        if dominant in (EvidenceDirection.ABSENT, EvidenceDirection.UNCERTAIN, EvidenceDirection.NEUTRAL):
            return base

        # EIE alignment bonus: only when EIE confirms the dominant direction
        eie = by_source.get(SourceType.FUNDAMENTAL_EIE)
        if eie and eie.direction == dominant:
            base = min(100.0, base + dfe_config.DFE_BONUS_EIE_ALIGNMENT)

        # MIA alignment bonus: only when MIA confirms the dominant direction
        mia = by_source.get(SourceType.AI_INTELLIGENCE)
        if mia and mia.direction == dominant:
            base = min(100.0, base + dfe_config.DFE_BONUS_MIA_ALIGNMENT)

        # Triple confirmation: ML + EIE + MIA all agree
        ml = by_source.get(SourceType.TECHNICAL_ML)
        if (
            ml  and ml.direction  == dominant
            and eie and eie.direction == dominant
            and mia and mia.direction == dominant
        ):
            base = min(100.0, base + dfe_config.DFE_BONUS_TRIPLE_CONFIRM)

        return base

    @staticmethod
    def _apply_agreement_multiplier(base: float, consensus: ConsensusLevel) -> float:
        multipliers = {
            ConsensusLevel.VERY_STRONG: 1.10,
            ConsensusLevel.STRONG:      1.00,
            ConsensusLevel.MODERATE:    0.90,
            ConsensusLevel.WEAK:        0.80,
        }
        return base * multipliers.get(consensus, 1.0)

    def _apply_conflict_penalty(self, base: float, conflict_score: float) -> float:
        if conflict_score > dfe_config.DFE_CONFIDENCE_CONFLICT_HIGH:
            return base - dfe_config.DFE_PENALTY_HIGH_CONFLICT
        if conflict_score > dfe_config.DFE_CONFIDENCE_CONFLICT_MED:
            return base - dfe_config.DFE_PENALTY_MED_CONFLICT
        return base

    def _apply_execution_risk_penalty(self, base: float, execution_risk: float) -> float:
        if execution_risk > dfe_config.DFE_CONFIDENCE_EXECRISK_HIGH:
            return base - dfe_config.DFE_PENALTY_HIGH_EXECRISK
        if execution_risk > dfe_config.DFE_CONFIDENCE_EXECRISK_MED:
            return base - dfe_config.DFE_PENALTY_MED_EXECRISK
        return base

    def _apply_ai_risk_penalty(self, base: float, ai_risk_level: str | None) -> float:
        if ai_risk_level == "CRITICAL":
            return base - dfe_config.DFE_PENALTY_AI_CRITICAL_RISK
        return base

    @staticmethod
    def _dominant_direction(items: List[EvidenceItem]) -> EvidenceDirection:
        weights: dict[EvidenceDirection, float] = {
            EvidenceDirection.BULLISH: 0.0,
            EvidenceDirection.BEARISH: 0.0,
            EvidenceDirection.NEUTRAL: 0.0,
        }
        for item in items:
            if item.direction in weights:
                weights[item.direction] += item.reliability * item.importance
        best = max(weights, key=weights.get)
        return best if weights[best] > 0.0 else EvidenceDirection.NEUTRAL
