"""
Decision Fusion Engine — central orchestrator of the DFE pipeline.

Lifecycle: startup() → background APScheduler cycle → shutdown()

The engine:
  1. Accepts a FusionInput bundle from callers (API endpoints, prediction service)
  2. Runs the full pipeline:
       Evidence → Agreement → Confidence → Rules → Recommendation → Explanation
  3. Assembles and stores a versioned DecisionObject
  4. Runs a background cycle to re-process when new intelligence is available
  5. Exposes health(), get_current(), and get_history()
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from decision_fusion.agreement_engine.calculator import AgreementCalculator
from decision_fusion.confidence_engine.fusion import ConfidenceFusion
from decision_fusion.evidence_engine.collector import EvidenceCollector
from decision_fusion.explanation_engine.builder import ExplanationBuilder
from decision_fusion.models.enums import MarketBiasEnum
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.recommendation_cache.cache import decision_cache
from decision_fusion.recommendation_engine.generator import RecommendationGenerator
from decision_fusion.rule_engine.evaluator import RuleEvaluator
from decision_fusion.schema.decision_object import DecisionObject
from decision_fusion.utils.config import dfe_config
from decision_fusion.utils.logger import logger
from decision_fusion.utils.metrics import DFEMetrics


class DecisionFusionEngine:
    """
    The central decision-making component of MFIP.

    Combines ML, EIE, and MIA intelligence into one explainable, versioned
    recommendation. Every output is deterministic and reproducible.

    Usage:
        engine = DecisionFusionEngine()
        await engine.startup()
        decision = await engine.process(fusion_input)
        await engine.shutdown()
    """

    def __init__(self) -> None:
        self._collector    = EvidenceCollector()
        self._agreement    = AgreementCalculator()
        self._confidence   = ConfidenceFusion()
        self._evaluator    = RuleEvaluator()
        self._generator    = RecommendationGenerator()
        self._explainer    = ExplanationBuilder()
        self._metrics      = DFEMetrics()
        self._scheduler:   Optional[AsyncIOScheduler] = None
        self._running:     bool = False
        self._started_at:  Optional[datetime] = None
        self._last_input:  Optional[FusionInput] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        logger.info("DFE starting up…")
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._scheduler.add_job(
            self._cycle,
            trigger          = "interval",
            seconds          = dfe_config.DFE_CYCLE_SECONDS,
            id               = "dfe_cycle",
            max_instances    = 1,
            coalesce         = True,
            replace_existing = True,
        )
        self._scheduler.start()
        self._running    = True
        self._started_at = datetime.now(timezone.utc)
        logger.info("DFE ready — cycle every {}s", dfe_config.DFE_CYCLE_SECONDS)

    async def shutdown(self) -> None:
        logger.info("DFE shutting down…")
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("DFE stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at

    # ── Core pipeline ──────────────────────────────────────────────────────────

    async def process(self, fusion_input: FusionInput) -> DecisionObject:
        """
        Run the full Decision Fusion pipeline on the given input bundle.

        Returns a fully populated, versioned DecisionObject.
        Always succeeds — worst case returns a WAIT recommendation with
        explanation noting what was unavailable.
        """
        t0 = time.monotonic()
        self._last_input = fusion_input
        now = fusion_input.current_time

        try:
            decision = self._run_pipeline(fusion_input, now)
        except Exception as exc:
            logger.error("DFE pipeline error: {} — returning WAIT fallback", exc)
            decision = self._fallback_decision(now, str(exc))

        await decision_cache.store(decision)

        elapsed_ms = (time.monotonic() - t0) * 1000
        await self._metrics.record(elapsed_ms, decision.recommendation)
        logger.info(
            "DFE: {} {} conf={:.1f} agreement={:.0f} conflict={:.0f} ({:.0f}ms)",
            decision.recommendation,
            decision.recommendation_strength,
            decision.decision_confidence,
            decision.agreement_score,
            decision.conflict_score,
            elapsed_ms,
        )
        return decision

    def _run_pipeline(self, fi: FusionInput, now: datetime) -> DecisionObject:
        # ── Stage 1: Evidence Collection ───────────────────────────────────────
        evidence_items = self._collector.collect(fi)

        # ── Stage 2: Agreement Engine ──────────────────────────────────────────
        agreement = self._agreement.compute(evidence_items)

        # ── Stage 3: Confidence Fusion ─────────────────────────────────────────
        preliminary_conf = self._confidence.compute(evidence_items, agreement, fi)

        # ── Stage 4: Rule Engine ───────────────────────────────────────────────
        context  = self._evaluator.build_context(evidence_items, agreement, preliminary_conf, fi)
        rule_eval = self._evaluator.evaluate(context)

        # ── Stage 5: Recommendation Generation ────────────────────────────────
        rec_result = self._generator.generate(evidence_items, agreement, preliminary_conf, rule_eval)

        # ── Stage 6: Supplementary computations ───────────────────────────────
        tech_align  = self._generator.compute_technical_alignment(fi)
        fund_align  = self._generator.compute_fundamental_alignment(fi)
        dominant    = RecommendationGenerator._dominant_direction(evidence_items)
        market_bias = self._generator.compute_market_bias(fi, dominant)
        expires_at  = self._generator.compute_expiry(rec_result.recommendation, rec_result.strength, now)

        # ── Stage 7: Explanation ───────────────────────────────────────────────
        explanation = self._explainer.build(
            evidence_items   = evidence_items,
            agreement_result = agreement,
            rule_evaluation  = rule_eval,
            recommendation   = rec_result.recommendation,
            confidence       = rec_result.confidence,
            fusion_input     = fi,
        )

        # ── Stage 8: Assemble DecisionObject ──────────────────────────────────
        return DecisionObject(
            decision_schema_version  = dfe_config.DECISION_SCHEMA_VERSION,
            recommendation           = rec_result.recommendation,
            recommendation_strength  = rec_result.strength,
            decision_confidence      = round(rec_result.confidence, 2),
            agreement_score          = agreement.agreement_score,
            conflict_score           = agreement.conflict_score,
            consensus_level          = agreement.consensus_level,
            technical_alignment      = tech_align,
            fundamental_alignment    = fund_align,
            market_bias              = market_bias,
            primary_reasons          = explanation.primary_reasons,
            supporting_evidence      = explanation.supporting_evidence,
            conflicting_reasons      = explanation.conflicting_reasons,
            confidence_drivers       = explanation.confidence_drivers,
            risk_factors             = explanation.risk_factors,
            generated_at             = now,
            expires_at               = expires_at,
            has_ml                   = fi.ml_prediction is not None,
            has_eie                  = bool(fi.eie_reports),
            has_mia                  = fi.mia_output is not None,
        )

    # ── Background cycle ───────────────────────────────────────────────────────

    async def _cycle(self) -> None:
        """
        Background cycle: re-process if the current decision is expired and
        we have a cached input from the last process() call.
        """
        if not decision_cache.is_expired():
            logger.debug("DFE cycle: decision still valid — skipping recompute")
            return

        if self._last_input is None:
            logger.debug("DFE cycle: no cached input — skipping")
            return

        logger.debug("DFE cycle: decision expired — reprocessing with last input")
        try:
            # Refresh timestamp before reprocessing
            import dataclasses
            fresh_input = dataclasses.replace(
                self._last_input,
                current_time=datetime.now(timezone.utc),
            )
            await self.process(fresh_input)
        except Exception as exc:
            logger.error("DFE background cycle failed: {}", exc)

    # ── Public read interface ──────────────────────────────────────────────────

    @property
    def current_decision(self) -> Optional[DecisionObject]:
        return decision_cache.current

    def get_history(self, limit: int = 20) -> list[DecisionObject]:
        return decision_cache.get_history(limit)

    def health(self) -> dict:
        current = decision_cache.current
        metrics = self._metrics.snapshot()
        return {
            "status":                 "operational" if self._running else "offline",
            "running":                self._running,
            "schema_version":         dfe_config.DECISION_SCHEMA_VERSION,
            "current_recommendation": current.recommendation if current else None,
            "recommendation_strength": current.recommendation_strength if current else None,
            "recommendation_age_s":   decision_cache.age_seconds(),
            "time_until_expiry_s":    decision_cache.seconds_until_expiry(),
            "is_expired":             decision_cache.is_expired(),
            "agreement_score":        current.agreement_score if current else None,
            "conflict_score":         current.conflict_score if current else None,
            "decision_confidence":    current.decision_confidence if current else None,
            "cache_size":             decision_cache.size(),
            "avg_processing_ms":      metrics.get("avg_processing_ms"),
            "total_decisions":        metrics.get("total_decisions"),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_decision(now: datetime, error: str) -> DecisionObject:
        from datetime import timedelta
        from decision_fusion.models.enums import (
            ConsensusLevel, Recommendation, RecommendationStrength,
        )
        return DecisionObject(
            recommendation          = Recommendation.WAIT,
            recommendation_strength = RecommendationStrength.WEAK,
            decision_confidence     = 0.0,
            agreement_score         = 0.0,
            conflict_score          = 0.0,
            consensus_level         = ConsensusLevel.WEAK,
            technical_alignment     = 0.0,
            fundamental_alignment   = 0.0,
            market_bias             = MarketBiasEnum.UNCERTAIN,
            primary_reasons         = [f"DFE pipeline error: {error}"],
            supporting_evidence     = [],
            conflicting_reasons     = [],
            confidence_drivers      = [],
            risk_factors            = ["Decision Fusion Engine encountered an unexpected error."],
            generated_at            = now,
            expires_at              = now + timedelta(seconds=dfe_config.DFE_EXPIRY_WAIT_S),
        )
