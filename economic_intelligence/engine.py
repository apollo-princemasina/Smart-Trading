"""
EconomicIntelligenceEngine — main orchestrator for the EIE pipeline.

Lifecycle:
  startup()  — warm cache from FF connector + start APScheduler job
  shutdown() — stop scheduler
  run_cycle() — process all MFIPEvents → EconomicIntelligenceReports → cache

The engine reads exclusively from the ForexFactory connector cache.
No network calls; no external dependencies beyond the connector.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from market_intel.models.enums import ImpactLevel, EventStatus
from market_intel.models.event import MFIPEvent

from economic_intelligence.event_classifier.classifier import EventClassifier
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.surprise_engine.calculator import SurpriseCalculator
from economic_intelligence.surprise_engine.models import SurpriseClass, SurpriseDirection
from economic_intelligence.direction_engine.rule_engine import DirectionRuleEngine
from economic_intelligence.direction_engine.models import EconomicDirection
from economic_intelligence.impact_engine.calculator import ImpactCalculator
from economic_intelligence.decay_engine.calculator import DecayCalculator
from economic_intelligence.execution_risk.calculator import ExecutionRiskCalculator
from economic_intelligence.intelligence_models.models import EconomicIntelligenceReport
from economic_intelligence.intelligence_cache.cache import intelligence_cache
from economic_intelligence.utils.config import eie_config
from economic_intelligence.utils.logger import logger


def _make_report_id(event_id: str, generated_at: datetime) -> str:
    raw = f"{event_id}|{generated_at.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _minutes_to_event(timestamp_utc: Optional[datetime], now: datetime) -> Optional[float]:
    if timestamp_utc is None:
        return None
    delta = (timestamp_utc - now).total_seconds() / 60.0
    return round(delta, 1)


def _overall_confidence(
    direction_confidence: float,
    event_type: EventType,
    has_surprise: bool,
) -> float:
    base = direction_confidence
    if has_surprise:
        base = min(1.0, base * 1.1)
    if event_type in (EventType.UNKNOWN, EventType.POLITICAL):
        base *= 0.5
    return round(max(0.0, min(1.0, base)), 3)


class EconomicIntelligenceEngine:
    """
    Orchestrates the full EIE pipeline on a configurable schedule.

    Usage:
        engine = EconomicIntelligenceEngine()
        await engine.startup()
        # ... FastAPI serves traffic ...
        await engine.shutdown()
    """

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._started_at: Optional[datetime] = None
        self._cycle_failures: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        logger.info("EIE starting up...")
        await self.run_cycle()   # warm cache immediately
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._scheduler.add_job(
            self.run_cycle,
            trigger="interval",
            seconds=eie_config.EIE_CYCLE_SECONDS,
            id="eie_cycle",
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        self._started_at = datetime.now(timezone.utc)
        logger.info("EIE ready — cycle every {}s", eie_config.EIE_CYCLE_SECONDS)

    async def shutdown(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("EIE shut down")

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    @property
    def started_at(self) -> Optional[datetime]:
        return self._started_at

    # ── Pipeline ──────────────────────────────────────────────────────────────

    async def run_cycle(self) -> None:
        """
        Full EIE cycle:
        1. Fetch events from FF connector cache (all three weeks)
        2. Process each event through the pipeline
        3. Build execution context
        4. Write to intelligence_cache
        """
        try:
            events = await self._fetch_all_events()
            if not events:
                logger.debug("EIE cycle: connector cache is empty — skipping")
                return

            now = datetime.now(timezone.utc)
            reports = [self._process_event(e, now) for e in events]

            # Build execution context from the freshly computed reports
            upcoming_high = [
                r for r in reports
                if (
                    not r.is_released
                    and r.importance == ImpactLevel.HIGH
                    and r.time_to_event is not None
                    and 0 <= r.time_to_event <= eie_config.EIE_RISK_LOOKAHEAD_MIN
                )
            ]
            active = [
                r for r in reports
                if r.is_released and r.remaining_influence > eie_config.EIE_ACTIVE_THRESHOLD
            ]

            is_market_open = self._is_market_open(now)
            is_holiday = any(r.importance == ImpactLevel.HOLIDAY for r in reports)

            context = ExecutionRiskCalculator.compute(
                upcoming_high=upcoming_high,
                active_events=active,
                is_market_open=is_market_open,
                is_holiday=is_holiday,
            )

            # Annotate per-event execution scores from the global context
            for r in reports:
                r.execution_risk = context.execution_risk
                r.execution_readiness = context.execution_readiness

            await intelligence_cache.set_reports(reports, context)
            self._cycle_failures = 0
            logger.debug("EIE cycle complete — {} events processed", len(reports))

        except Exception as exc:
            self._cycle_failures += 1
            logger.error("EIE cycle failed (#{}) — {}", self._cycle_failures, exc)

    def _process_event(self, event: MFIPEvent, now: datetime) -> EconomicIntelligenceReport:
        # 1 — Classify
        event_type = EventClassifier.classify(event)

        # 2 — Surprise
        surprise = SurpriseCalculator.compute(event, event_type)

        # 3 — Direction
        signal = DirectionRuleEngine.resolve(event, event_type, surprise)

        # 4 — Impact score
        impact_score = ImpactCalculator.compute(event, event_type, surprise)

        # 5 — Decay
        remaining_influence, event_age_hours = DecayCalculator.compute(
            event, event_type, impact_score, now
        )

        # 6 — Time to event
        time_to_event = _minutes_to_event(event.timestamp_utc, now)

        # 7 — Is released
        is_released = event.status != EventStatus.SCHEDULED

        # 8 — Confidence
        confidence = _overall_confidence(
            signal.confidence, event_type, surprise is not None
        )

        generated_at = now
        report_id = _make_report_id(event.event_id, generated_at)

        return EconomicIntelligenceReport(
            report_id=report_id,
            event_id=event.event_id,
            generated_at=generated_at,
            event_title=event.title,
            currency=event.currency,
            country=event.country,
            timestamp_utc=event.timestamp_utc,
            is_released=is_released,
            importance=event.impact,
            event_type=event_type,
            impact_score=round(impact_score, 2),
            surprise=round(surprise.raw_surprise, 6) if surprise else None,
            pct_surprise=round(surprise.pct_surprise, 4) if (surprise and surprise.pct_surprise is not None) else None,
            surprise_class=surprise.surprise_class if surprise else SurpriseClass.NONE,
            surprise_direction=surprise.direction if surprise else SurpriseDirection.IN_LINE,
            economic_direction=signal.direction,
            direction_confidence=round(signal.confidence, 3),
            direction_rationale=signal.rationale,
            remaining_influence=remaining_influence,
            event_age_hours=event_age_hours,
            time_to_event=time_to_event,
            execution_risk=0.0,      # filled after context is built
            execution_readiness=0.0,
            confidence=confidence,
            last_updated=generated_at,
        )

    @staticmethod
    async def _fetch_all_events() -> list[MFIPEvent]:
        """Pull events from the FF connector cache (all three weeks)."""
        from forex_factory_connector.cache.memory_cache import connector_cache

        events: list[MFIPEvent] = []
        for week in ("thisweek", "nextweek", "lastweek"):
            try:
                entry = await connector_cache.get_calendar(week)
                if entry:
                    events.extend(entry.events)
            except Exception:
                pass
        return events

    @staticmethod
    def _is_market_open(now: datetime) -> bool:
        """
        Forex is open Sun 22:00 UTC – Fri 22:00 UTC.
        Weekday 5 = Saturday, weekday 6 = Sunday.
        """
        weekday = now.weekday()
        hour = now.hour

        if weekday == 5:    # Saturday — fully closed
            return False
        if weekday == 6:    # Sunday — opens at 22:00 UTC
            return hour >= 22
        if weekday == 4:    # Friday — closes at 22:00 UTC
            return hour < 22
        return True         # Mon–Thu always open
