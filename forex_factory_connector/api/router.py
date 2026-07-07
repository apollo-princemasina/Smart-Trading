"""
Intelligence Layer router — mounted at /api/v1/intelligence by the MFIP v1 router.

Route map
---------
  GET /intelligence/calendar                   All events for a given week
  GET /intelligence/calendar/high-impact       HIGH-impact events for a week

  GET /intelligence/events/today               Today's events
  GET /intelligence/events/high-impact         HIGH-impact events (this week default)
  GET /intelligence/events/next                Next single upcoming event

  GET /intelligence/speeches                   This week's speech events
  GET /intelligence/news                       Phase 3 stub — 503
  GET /intelligence/sentiment                  Phase 3 stub — 503

  GET /intelligence/health                     Operational dashboard
"""
from fastapi import APIRouter

from .endpoints import calendar, events, speeches, news, sentiment, health as health_ep

intelligence_router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

intelligence_router.include_router(calendar.router)
intelligence_router.include_router(events.router)
intelligence_router.include_router(speeches.router)
intelligence_router.include_router(news.router)
intelligence_router.include_router(sentiment.router)
intelligence_router.include_router(health_ep.router)
