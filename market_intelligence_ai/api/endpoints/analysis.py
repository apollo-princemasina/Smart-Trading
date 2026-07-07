"""
Analysis endpoints — trigger analysis and retrieve results.

POST /intelligence/analyse/event      — analyse a released economic event
POST /intelligence/analyse/headline   — analyse a market headline
GET  /intelligence/analyses           — list recent AI analyses
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from market_intelligence_ai.api.schemas import (
    AnalyseEventRequest,
    AnalyseHeadlineRequest,
    AnalysisResponse,
    AnalysisListResponse,
)
from market_intelligence_ai.market_context_compiler.context_models import (
    EventTrigger,
    HeadlineTrigger,
    EIESnapshot,
)
from market_intelligence_ai.utils.logger import logger

router = APIRouter()


def _get_engine(request: Request):
    engine = getattr(request.app.state, "mia", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Market Intelligence AI Engine not running.")
    return engine


@router.post("/analyse/event", response_model=AnalysisResponse, summary="Analyse economic event")
async def analyse_event(body: AnalyseEventRequest, request: Request):
    """
    Trigger AI analysis for a released economic event.

    The engine builds a rich context from the event data plus the current EIE state,
    then calls the Market Intelligence Agent to produce structured intelligence.
    """
    engine = _get_engine(request)

    trigger = EventTrigger(
        event_id           = body.event_id,
        title              = body.title,
        currency           = body.currency,
        timestamp          = datetime.now(timezone.utc),
        importance         = body.importance,
        forecast           = body.forecast,
        actual             = body.actual,
        previous           = body.previous,
        surprise_class     = body.surprise_class,
        surprise_direction = body.surprise_direction,
        economic_direction = body.economic_direction,
    )

    try:
        eie_snapshot = await engine._build_eie_snapshot()
    except Exception:
        eie_snapshot = EIESnapshot()

    payload  = engine._context_builder.build_for_event(trigger, eie_snapshot)
    analysis = await engine.agent.analyze(payload)
    await engine._store(analysis)

    return AnalysisResponse(
        analysis   = analysis,
        request_id = str(uuid.uuid4()),
        timestamp  = datetime.now(timezone.utc),
    )


@router.post("/analyse/headline", response_model=AnalysisResponse, summary="Analyse market headline")
async def analyse_headline(body: AnalyseHeadlineRequest, request: Request):
    """
    Trigger AI analysis for a market headline.

    The engine builds context from the headline plus current EIE state,
    then calls the Market Intelligence Agent.
    """
    engine = _get_engine(request)

    trigger = HeadlineTrigger(
        headline_id         = body.headline_id,
        headline            = body.headline,
        source              = body.source,
        timestamp           = datetime.now(timezone.utc),
        affected_currencies = body.affected_currencies,
    )

    try:
        eie_snapshot = await engine._build_eie_snapshot()
    except Exception:
        eie_snapshot = EIESnapshot()

    payload  = engine._context_builder.build_for_headline(trigger, eie_snapshot)
    analysis = await engine.agent.analyze(payload)
    await engine._store(analysis)

    return AnalysisResponse(
        analysis   = analysis,
        request_id = str(uuid.uuid4()),
        timestamp  = datetime.now(timezone.utc),
    )


@router.get("/analyses", response_model=AnalysisListResponse, summary="List recent AI analyses")
async def list_analyses(request: Request, limit: int = 20):
    """Return the most recent N market intelligence analyses."""
    engine   = _get_engine(request)
    analyses = await engine.get_recent_analyses(limit=min(limit, 100))
    return AnalysisListResponse(analyses=analyses, total=len(analyses))
