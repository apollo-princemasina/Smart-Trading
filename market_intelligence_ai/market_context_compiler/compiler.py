"""
MarketContextCompiler — the heart of the Market Intelligence AI Layer.

Responsibility: Assemble all available market information into a single, rich
ContextPayload that the Market Intelligence Agent will reason from autonomously.

This is Context Engineering, not Prompt Engineering. Instead of selecting
different prompts for CPI vs NFP vs speeches, the MarketContextCompiler gives
the AI rich, structured, deterministic context and lets it reason.

The MarketContextCompiler NEVER calls the AI. It only prepares and structures data.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from market_intelligence_ai.market_context_compiler.context_models import (
    ContextPayload,
    EventTrigger,
    HeadlineTrigger,
    EIESnapshot,
)
from market_intelligence_ai.models.enums import AnalysisType
from market_intelligence_ai.utils.config import mia_config
from market_intelligence_ai.utils.logger import logger


def _detect_session(now: datetime) -> str:
    """Determine the current forex market session by UTC hour."""
    hour = now.hour
    if 13 <= hour < 17:
        return "OVERLAP"       # London + New York overlap
    elif mia_config.MIA_NEW_YORK_START_UTC <= hour < 22:
        return "NEW_YORK"
    elif mia_config.MIA_LONDON_START_UTC <= hour < mia_config.MIA_NEW_YORK_START_UTC:
        return "LONDON"
    elif hour >= 22 or hour < mia_config.MIA_ASIA_START_UTC + 8:
        return "ASIA"
    else:
        return "OFF_MARKET"


def _format_event_trigger(event: EventTrigger) -> str:
    lines = [
        "--- ECONOMIC EVENT ---",
        f"Title:      {event.title}",
        f"Currency:   {event.currency}",
        f"Time:       {event.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Importance: {event.importance}",
    ]
    if event.previous is not None:
        lines.append(f"Previous:   {event.previous}")
    if event.forecast is not None:
        lines.append(f"Forecast:   {event.forecast}")
    if event.actual is not None:
        lines.append(f"Actual:     {event.actual}")
    if event.surprise_class != "NONE":
        lines.append(f"Surprise:   {event.surprise_class} ({event.surprise_direction})")
    lines.append(f"EIE Direction: {event.economic_direction}")
    return "\n".join(lines)


def _format_headline_trigger(headline: HeadlineTrigger) -> str:
    currencies = ", ".join(headline.affected_currencies) or "Unknown"
    return (
        "--- MARKET HEADLINE ---\n"
        f"Headline:            {headline.headline}\n"
        f"Source:              {headline.source}\n"
        f"Published:           {headline.timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Affected Currencies: {currencies}"
    )


def _format_eie_snapshot(snap: EIESnapshot, max_active: int, max_upcoming: int) -> str:
    lines = ["--- ECONOMIC INTELLIGENCE CONTEXT ---"]

    if snap.dominant_directions:
        directions = ", ".join(f"{ccy}: {bias}" for ccy, bias in snap.dominant_directions.items())
        lines.append(f"Dominant Directions: {directions}")

    lines.append(f"Execution Risk:      {snap.execution_risk:.0f}/100")
    lines.append(f"Execution Readiness: {snap.execution_readiness:.0f}/100")

    active = snap.active_events[:max_active]
    if active:
        lines.append(f"Active Events ({len(active)} of {len(snap.active_events)}):")
        for e in active:
            lines.append(
                f"  [{e.get('currency', '?')}] {e.get('title', '?')} — "
                f"{e.get('economic_direction', '?')} "
                f"(impact={e.get('impact_score', 0):.0f}, "
                f"influence={e.get('remaining_influence', 0):.0f}%)"
            )

    upcoming = snap.upcoming_high_impact[:max_upcoming]
    if upcoming:
        lines.append(f"Upcoming High-Impact ({len(upcoming)} shown):")
        for e in upcoming:
            lines.append(
                f"  [{e.get('currency', '?')}] {e.get('title', '?')} "
                f"at {e.get('event_time', '?')}"
            )

    if snap.snapshot_at:
        lines.append(f"Context captured at: {snap.snapshot_at.strftime('%Y-%m-%d %H:%M UTC')}")

    return "\n".join(lines)


class MarketContextCompiler:
    """
    Assembles ContextPayload objects from all available data sources.

    This is the core of Context Engineering. The compiler automatically gathers
    and structures everything the AI needs to reason from — no manual routing,
    no event-type-specific logic.

    Usage:
        compiler = MarketContextCompiler()

        # For an economic event:
        payload = compiler.build_for_event(event_trigger, eie_snapshot)

        # For a market headline:
        payload = compiler.build_for_headline(headline_trigger, eie_snapshot)

        # Format as the AI user message:
        user_message = compiler.format_as_user_message(payload)

        # Get the deterministic cache key:
        key = MarketContextCompiler.cache_key(payload, provider_version="groq_v1")
    """

    def build_for_event(
        self,
        event:        EventTrigger,
        eie_snapshot: Optional[EIESnapshot] = None,
        now:          Optional[datetime] = None,
    ) -> ContextPayload:
        now = now or datetime.now(timezone.utc)
        payload = ContextPayload(
            analysis_type      = AnalysisType.EVENT,
            primary_currency   = event.currency,
            analysis_timestamp = now,
            current_session    = _detect_session(now),
            event_trigger      = event,
            eie_snapshot       = eie_snapshot or EIESnapshot(),
        )
        payload.validate()
        logger.debug(
            "MarketContextCompiler: built EVENT payload for {} ({})",
            event.event_id, event.currency,
        )
        return payload

    def build_for_headline(
        self,
        headline:     HeadlineTrigger,
        eie_snapshot: Optional[EIESnapshot] = None,
        now:          Optional[datetime] = None,
    ) -> ContextPayload:
        now = now or datetime.now(timezone.utc)
        primary = headline.affected_currencies[0] if headline.affected_currencies else "USD"
        payload = ContextPayload(
            analysis_type      = AnalysisType.HEADLINE,
            primary_currency   = primary,
            analysis_timestamp = now,
            current_session    = _detect_session(now),
            headline_trigger   = headline,
            eie_snapshot       = eie_snapshot or EIESnapshot(),
        )
        payload.validate()
        logger.debug(
            "MarketContextCompiler: built HEADLINE payload for {}",
            headline.headline_id,
        )
        return payload

    def build_for_general_context(
        self,
        eie_snapshot: Optional[EIESnapshot] = None,
        now: Optional[datetime] = None,
        market_ctx: Optional[dict] = None,
    ) -> ContextPayload:
        """Build a ContextPayload for a periodic general market narrative with no specific trigger."""
        now = now or datetime.now(timezone.utc)
        payload = ContextPayload(
            analysis_type      = AnalysisType.GENERAL,
            primary_currency   = "EURUSD",
            analysis_timestamp = now,
            current_session    = _detect_session(now),
            eie_snapshot       = eie_snapshot or EIESnapshot(),
            market_ctx         = market_ctx or {},
        )
        payload.validate()
        logger.debug("MarketContextCompiler: built GENERAL payload (session={})", payload.current_session)
        return payload

    def format_as_user_message(self, payload: ContextPayload) -> str:
        """
        Format a ContextPayload into a structured user message for the AI.

        The AI receives this as its user message alongside the fixed system prompt.
        All intelligence context is supplied here — the system prompt never changes.
        """
        sections: list[str] = []

        sections.append(
            f"ANALYSIS REQUEST\n"
            f"Timestamp:        {payload.analysis_timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Primary Currency: {payload.primary_currency}\n"
            f"Market Session:   {payload.current_session}"
        )

        if payload.analysis_type == AnalysisType.GENERAL:
            ctx = payload.market_ctx
            lines = [
                "ANALYSIS TYPE: General Market Assessment — EURUSD",
                "Provide a comprehensive institutional overview of current EURUSD conditions, "
                "explicitly referencing any upcoming EUR/USD news events and their expected impact on price.",
            ]
            if ctx:
                lines.append("--- LIVE MARKET DATA ---")
                if ctx.get("current_price"):
                    lines.append(f"Current Price (EURUSD): {ctx['current_price']:.5f}")
                if ctx.get("regime"):
                    lines.append(f"Market Regime:          {ctx['regime']}")
                if ctx.get("atr_pips"):
                    lines.append(f"ATR (pips):             {ctx['atr_pips']:.1f}")
                if ctx.get("adx") is not None:
                    lines.append(f"ADX:                    {ctx['adx']:.1f}")
                if ctx.get("regime_narrative"):
                    lines.append(f"Regime Signal:          {ctx['regime_narrative']}")
                if ctx.get("session"):
                    lines.append(f"Trading Session:        {ctx['session']}")
                if ctx.get("latest_direction"):
                    direction = ctx['latest_direction']
                    conf      = ctx.get('latest_confidence', 0) * 100
                    pb        = (ctx.get('prob_buy',  0) or 0) * 100
                    ps        = (ctx.get('prob_sell', 0) or 0) * 100
                    ph        = (ctx.get('prob_hold', 0) or 0) * 100
                    lines.append(
                        f"ML Signal (latest):     {direction} @ {conf:.0f}% confidence  "
                        f"[BUY={pb:.0f}% SELL={ps:.0f}% HOLD={ph:.0f}%]"
                    )
                # Upcoming EUR/USD news from Forex Factory
                ff_news = ctx.get("ff_upcoming_news") or []
                if ff_news:
                    lines.append("--- UPCOMING EUR/USD NEWS (Forex Factory, next 12h) ---")
                    for ev in ff_news:
                        impact = ev.get("impact", "").upper()
                        flag   = "!!!" if impact == "HIGH" else "!!" if impact == "MEDIUM" else "!"
                        line   = f"  [{flag}] {ev.get('currency','')} | {ev.get('title','')} @ {ev.get('time','')}"
                        if ev.get("forecast"): line += f"  forecast={ev['forecast']}"
                        if ev.get("previous"): line += f"  prev={ev['previous']}"
                        if ev.get("actual"):   line += f"  actual={ev['actual']}"
                        lines.append(line)
                    lines.append(
                        "Assess how these scheduled releases may affect EURUSD directional bias "
                        "and execution risk in the current session."
                    )
            sections.append("\n".join(lines))

        if payload.event_trigger is not None:
            sections.append(_format_event_trigger(payload.event_trigger))

        if payload.headline_trigger is not None:
            sections.append(_format_headline_trigger(payload.headline_trigger))

        sections.append(_format_eie_snapshot(
            payload.eie_snapshot,
            max_active   = mia_config.MIA_CONTEXT_MAX_ACTIVE_EVENTS,
            max_upcoming = mia_config.MIA_CONTEXT_MAX_UPCOMING_EVENTS,
        ))

        sections.append(
            "Apply your five institutional reasoning perspectives and produce your JSON analysis now."
        )

        return "\n\n".join(sections)

    @staticmethod
    def cache_key(
        payload:          ContextPayload,
        provider_version: str = mia_config.PROVIDER_VERSION,
    ) -> str:
        """
        Deterministic cache key derived from all context-identifying fields.

        Includes: context_schema_version, analysis_schema_version, provider_version,
        analysis_type, primary_currency, event_id / surprise_class, headline_hash.

        Any version bump automatically invalidates all stale cache entries.
        """
        parts: list[str] = [
            payload.context_schema_version,
            mia_config.ANALYSIS_SCHEMA_VERSION,
            provider_version,
            payload.analysis_type.value,
            payload.primary_currency,
        ]

        if payload.event_trigger:
            parts.append(f"event:{payload.event_trigger.event_id}")
            parts.append(f"surprise:{payload.event_trigger.surprise_class}")

        if payload.headline_trigger:
            h = hashlib.sha256(
                payload.headline_trigger.headline.strip().lower().encode()
            ).hexdigest()[:12]
            parts.append(f"headline:{h}")

        if payload.analysis_type == AnalysisType.GENERAL and payload.market_ctx:
            # Include regime + truncated price in cache key so different market states vary
            regime = payload.market_ctx.get("regime", "")
            price_bucket = round(payload.market_ctx.get("current_price", 0) * 1000)  # 1.143xx → 1143
            parts.append(f"regime:{regime}:price:{price_bucket}")

        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
