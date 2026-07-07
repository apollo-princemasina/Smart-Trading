import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from market_intel.models.enums import ImpactLevel

from ...fetcher.cdn_fetcher   import fetch_calendar
from ...parser.calendar_parser import parse_calendar_json
from ...validator.schema_validator import validate_and_build_events
from ...validator.data_quality     import check_quality
from ...cache.memory_cache         import connector_cache
from ...cache.etag_store           import etag_store
from ...utils.config               import settings
from ...utils.logger               import logger
from ..health_reporter             import health


def _disk_cache_path(week: str) -> Path:
    return Path(settings.DISK_CACHE_DIR) / f"{week}.json"


def _disk_cache_age_hours(week: str) -> float:
    """Return how old the disk cache is in hours, or inf if it doesn't exist."""
    p = _disk_cache_path(week)
    if not p.exists():
        return float("inf")
    age_s = time.time() - p.stat().st_mtime
    return age_s / 3600


def _load_from_disk(week: str) -> bytes | None:
    """Return raw JSON bytes from disk cache, or None if unavailable/expired."""
    p = _disk_cache_path(week)
    age_h = _disk_cache_age_hours(week)
    if age_h > settings.DISK_CACHE_TTL_HOURS:
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


def _save_to_disk(week: str, body: bytes) -> None:
    """Persist raw CDN response bytes to disk for restart recovery."""
    p = _disk_cache_path(week)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        logger.debug(f"[calendar:{week}] saved {len(body)} bytes to disk cache")
    except Exception as exc:
        logger.warning(f"[calendar:{week}] could not write disk cache — {exc}")


async def _populate_from_bytes(week: str, job_id: str, body: bytes, etag: str | None) -> int:
    """Parse → validate → populate memory cache. Returns event count."""
    raw    = parse_calendar_json(body)
    events = validate_and_build_events(raw, source_week=week)
    check_quality(events)
    await connector_cache.set_calendar(week, events, etag=etag)
    return len(events)


async def run_calendar_job(week: str = "thisweek") -> None:
    """
    Full pipeline for one calendar week: fetch → parse → validate → cache.

    On startup or after a CDN 429, falls back to disk cache (up to 6 h old).
    After every successful CDN fetch, raw bytes are saved to disk so the next
    restart can skip the CDN entirely (returns 304 once ETags are persistent).

    Records timing and success/failure metrics in HealthReporter.
    Does not raise — errors stay inside the scheduler job boundary.
    """
    job_id = f"calendar:{week}"

    if health.is_open(job_id):
        logger.warning(f"[{job_id}] circuit open — skipping poll")
        return

    t0 = time.monotonic()
    try:
        # ── Try CDN first ────────────────────────────────────────────────────
        current_etag = etag_store.get(job_id)
        result       = await fetch_calendar(week, current_etag=current_etag)

        if result.not_modified:
            # 304 — data unchanged; ETags survived restart → no re-fetch needed.
            health.record_success(job_id, response_time_ms=(time.monotonic() - t0) * 1000)
            logger.debug(f"[{job_id}] 304 not modified — disk ETags working")
            return

        if result.not_found:
            # 404 — CDN hasn't published this week yet (normal for lastweek/nextweek).
            # Record as success so the circuit breaker doesn't open.
            health.record_success(job_id, response_time_ms=(time.monotonic() - t0) * 1000)
            logger.debug(f"[{job_id}] 404 not published — skipping (not a failure)")
            return

        if result.rate_limited:
            # 429 — CDN is throttling us. Try to recover from disk cache.
            disk_body = _load_from_disk(week)
            if disk_body:
                age_h = _disk_cache_age_hours(week)
                count = await _populate_from_bytes(week, job_id, disk_body, current_etag)
                health.record_success(job_id, response_time_ms=(time.monotonic() - t0) * 1000)
                logger.warning(
                    f"[{job_id}] CDN 429 — served {count} events from disk cache "
                    f"({age_h:.1f}h old). Will retry on next poll."
                )
            else:
                logger.warning(
                    f"[{job_id}] CDN 429 and no disk cache available — "
                    "events will be empty until rate limit clears"
                )
                health.record_failure(job_id, Exception("CDN 429, no disk fallback"))
                await connector_cache.mark_stale(week)
            return

        # ── Successful CDN response ──────────────────────────────────────────
        etag_store.set(job_id, result.etag)
        _save_to_disk(week, result.body)

        count      = await _populate_from_bytes(week, job_id, result.body, result.etag)
        elapsed_ms = (time.monotonic() - t0) * 1000
        health.record_success(job_id, response_time_ms=elapsed_ms)
        logger.info(f"[{job_id}] updated: {count} events ({elapsed_ms:.0f} ms)")

    except Exception as exc:
        health.record_failure(job_id, exc)
        # Attempt disk cache recovery even on unexpected errors
        disk_body = _load_from_disk(week)
        if disk_body:
            try:
                count = await _populate_from_bytes(week, job_id, disk_body, etag_store.get(job_id))
                logger.warning(
                    f"[{job_id}] CDN error ({exc!r}) — recovered {count} events from disk cache"
                )
                return
            except Exception:
                pass
        await connector_cache.mark_stale(week)


def get_adaptive_interval() -> int:
    """Shorten poll interval to 60 s when a HIGH-impact event is within the lookahead window."""
    try:
        entry = connector_cache._weeks.get("thisweek")
        if entry is None:
            return settings.CALENDAR_POLL_SECONDS
        now      = datetime.now(timezone.utc)
        lookahead = timedelta(minutes=settings.HIGH_IMPACT_LOOKAHEAD_MINUTES)
        for event in entry.events:
            if (
                event.is_high_impact
                and event.timestamp_utc
                and timedelta(0) < (event.timestamp_utc - now) < lookahead
            ):
                return settings.HIGH_IMPACT_POLL_SECONDS
    except Exception:
        pass
    return settings.CALENDAR_POLL_SECONDS
