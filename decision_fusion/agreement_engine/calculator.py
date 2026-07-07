"""
Agreement Calculator — measures how strongly different intelligence sources agree.

Each pair of directional sources is compared. Agreement and conflict scores
are computed as weighted sums, so high-reliability sources have more influence
on the final scores than low-reliability sources.

Agreement Score (0–100): Higher means more inter-source agreement.
Conflict Score  (0–100): Higher means more inter-source contradiction.
"""
from __future__ import annotations

from typing import List

from decision_fusion.models.enums import ConsensusLevel, EvidenceDirection
from decision_fusion.models.evidence import AgreementResult, EvidenceItem
from decision_fusion.utils.logger import logger


class AgreementCalculator:
    """
    Computes agreement and conflict scores across all active evidence sources.

    Sources that are ABSENT or UNCERTAIN are excluded from pair comparisons
    (they contribute no directional information to agree or disagree about).
    NEUTRAL sources are included as a "no signal" position — two neutral sources
    agree with each other, but a neutral source neither agrees nor conflicts
    with a BULLISH or BEARISH source (treated as partial / non-overlapping).
    """

    def compute(self, evidence_items: List[EvidenceItem]) -> AgreementResult:
        """
        Compute agreement and conflict across the given evidence items.

        Returns an AgreementResult with:
          - agreement_score: 0–100 weighted agreement
          - conflict_score:  0–100 weighted conflict
          - consensus_level: derived from agreement_score
          - aligned_sources: list of source labels that agree on the dominant direction
          - conflicting_sources: list that oppose the dominant direction
          - neutral_sources: NEUTRAL items that carry no directional opinion
        """
        # Only consider sources with an actual directional position
        directional = [
            item for item in evidence_items
            if item.direction not in (EvidenceDirection.ABSENT, EvidenceDirection.UNCERTAIN)
        ]

        if len(directional) < 2:
            # Cannot compute agreement with fewer than 2 sources
            logger.debug("Agreement: insufficient directional sources ({})", len(directional))
            return self._no_agreement_result(directional)

        total_possible_weight = 0.0
        agreement_weight      = 0.0
        conflict_weight       = 0.0

        for i in range(len(directional)):
            for j in range(i + 1, len(directional)):
                a = directional[i]
                b = directional[j]

                # Pair weight = geometric mean of individual weights
                pair_weight = (a.reliability * a.importance) * (b.reliability * b.importance)
                total_possible_weight += pair_weight

                if a.direction == b.direction:
                    agreement_weight += pair_weight
                elif self._are_opposed(a.direction, b.direction):
                    conflict_weight += pair_weight
                # NEUTRAL paired with BULLISH/BEARISH: partial — counts as neither

        if total_possible_weight == 0.0:
            return self._no_agreement_result(directional)

        agreement_score = round((agreement_weight / total_possible_weight) * 100.0, 1)
        conflict_score  = round((conflict_weight  / total_possible_weight) * 100.0, 1)

        # Determine dominant direction for labelling
        dominant = self._dominant_direction(directional)
        aligned, conflicting, neutral = self._classify_sources(directional, dominant)

        consensus_level = self._score_to_consensus(agreement_score)

        logger.debug(
            "Agreement: score={:.1f}  conflict={:.1f}  consensus={}  dominant={}",
            agreement_score, conflict_score, consensus_level, dominant,
        )

        return AgreementResult(
            agreement_score     = agreement_score,
            conflict_score      = conflict_score,
            consensus_level     = consensus_level,
            aligned_sources     = aligned,
            conflicting_sources = conflicting,
            neutral_sources     = neutral,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _are_opposed(a: EvidenceDirection, b: EvidenceDirection) -> bool:
        """True only for a directly opposing pair (BULLISH ↔ BEARISH)."""
        opposites = {
            (EvidenceDirection.BULLISH, EvidenceDirection.BEARISH),
            (EvidenceDirection.BEARISH, EvidenceDirection.BULLISH),
        }
        return (a, b) in opposites

    @staticmethod
    def _dominant_direction(items: List[EvidenceItem]) -> EvidenceDirection:
        """Determine dominant direction via weighted vote."""
        weights: dict[EvidenceDirection, float] = {
            EvidenceDirection.BULLISH: 0.0,
            EvidenceDirection.BEARISH: 0.0,
            EvidenceDirection.NEUTRAL: 0.0,
        }
        for item in items:
            if item.direction in weights:
                weights[item.direction] += item.reliability * item.importance

        # Pick the highest, defaulting to NEUTRAL on ties
        best = EvidenceDirection.NEUTRAL
        best_w = 0.0
        for direction, w in weights.items():
            if w > best_w:
                best_w = w
                best = direction
        return best

    @staticmethod
    def _classify_sources(
        items: List[EvidenceItem],
        dominant: EvidenceDirection,
    ) -> tuple[list[str], list[str], list[str]]:
        aligned:     list[str] = []
        conflicting: list[str] = []
        neutral:     list[str] = []

        for item in items:
            if item.direction == EvidenceDirection.NEUTRAL:
                neutral.append(item.label)
            elif item.direction == dominant:
                aligned.append(item.label)
            else:
                conflicting.append(item.label)

        return aligned, conflicting, neutral

    @staticmethod
    def _score_to_consensus(score: float) -> ConsensusLevel:
        if score >= 80:
            return ConsensusLevel.VERY_STRONG
        if score >= 60:
            return ConsensusLevel.STRONG
        if score >= 40:
            return ConsensusLevel.MODERATE
        return ConsensusLevel.WEAK

    @staticmethod
    def _no_agreement_result(items: List[EvidenceItem]) -> AgreementResult:
        labels = [i.label for i in items]
        return AgreementResult(
            agreement_score     = 0.0,
            conflict_score      = 0.0,
            consensus_level     = ConsensusLevel.WEAK,
            aligned_sources     = labels,
            conflicting_sources = [],
            neutral_sources     = [],
        )
