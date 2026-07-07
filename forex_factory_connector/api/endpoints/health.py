from datetime import datetime, timezone
from fastapi import APIRouter, Request

from market_intel.models.enums import ImpactLevel

from ...cache.memory_cache        import connector_cache
from ...scheduler.health_reporter import health, SCHEMA_VERSION
from ...utils.config              import settings
from ..schemas import IntelligenceHealthOut, JobHealthOut

router = APIRouter(prefix="/health", tags=["Intelligence — Health"])


def _job_health(job_id: str, connector) -> JobHealthOut:
    m = health.get_job(job_id)
    if m is None:
        return JobHealthOut(
            job_id=job_id, status="unknown", poll_interval_s=0,
            last_success=None, last_failure=None, next_run=None,
            success_count=0, failure_count=0, retry_count=0,
            circuit_open=False, avg_response_ms=None, last_response_ms=None,
        )
    return JobHealthOut(
        job_id=job_id,
        status=m.status,
        poll_interval_s=m.poll_interval_s,
        last_success=m.last_success,
        last_failure=m.last_failure,
        next_run=connector.next_run(job_id) if connector else None,
        success_count=m.success_count,
        failure_count=m.failure_count,
        retry_count=m.retry_count,
        circuit_open=m.circuit_open,
        avg_response_ms=m.avg_response_ms,
        last_response_ms=m.last_response_ms,
    )


@router.get("", response_model=IntelligenceHealthOut, summary="Connector operational dashboard")
async def get_intelligence_health(request: Request) -> IntelligenceHealthOut:
    connector = getattr(request.app.state, "ff_connector", None)

    # Cache metrics
    cache_populated = {
        w: connector_cache.is_populated(w)
        for w in ("thisweek", "nextweek", "lastweek")
    }

    total_events = high_impact_events = speeches = 0
    for week in ("thisweek", "nextweek", "lastweek"):
        if connector_cache.is_populated(week):
            try:
                entry = await connector_cache.get_calendar(week)
                total_events   += len(entry.events)
                high_impact_events += sum(1 for e in entry.events if e.is_high_impact)
                speeches       += sum(1 for e in entry.events if e.is_speech)
            except Exception:
                pass

    # Per-job metrics
    job_ids = [
        "calendar:thisweek", "calendar:nextweek", "calendar:lastweek",
        "speeches", "news", "sentiment",
    ]
    jobs_out = {jid: _job_health(jid, connector) for jid in job_ids}

    return IntelligenceHealthOut(
        schema_version=SCHEMA_VERSION,
        provider="forex_factory",
        connector_status=health.overall_status if connector else "unavailable",
        scheduler_running=connector.is_running if connector else False,
        started_at=connector.started_at if connector else None,
        uptime_s=connector.uptime_s if connector else None,
        cache_populated=cache_populated,
        calendar_events_total=total_events,
        calendar_events_high_impact=high_impact_events,
        speeches_cached=speeches,
        news_items_cached=0,   # Phase 3
        jobs=jobs_out,
        calendar_poll_s=settings.CALENDAR_POLL_SECONDS,
        news_poll_s=settings.NEWS_POLL_SECONDS,
        sentiment_poll_s=settings.SENTIMENT_POLL_SECONDS,
        speeches_poll_s=settings.SPEECHES_POLL_SECONDS,
    )
