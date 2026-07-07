from datetime import datetime, timezone
from typing import Any

from market_intel.models.event import MFIPEvent
from market_intel.models.enums import ImpactLevel, EventStatus, EventCategory, Provider

from ..normalizer.time_normalizer    import parse_event_datetime
from ..normalizer.impact_normalizer  import normalize_impact
from ..normalizer.value_normalizer   import normalize_value
from ..normalizer.event_id           import build_event_id, build_provider_event_id
from ..normalizer.category_inferrer  import infer_category, infer_is_speech, currency_to_country
from ..utils.logger                  import logger


def _derive_status(actual: str | None) -> EventStatus:
    return EventStatus.RELEASED if actual is not None else EventStatus.SCHEDULED


def validate_and_build_events(
    raw_events: list[dict[str, Any]],
    source_week: str,
) -> list[MFIPEvent]:
    """
    Convert raw CDN dicts into canonical MFIPEvent objects.

    Handles both JSON shapes from the CDN:
      JSON: date = ISO 8601 timestamp (no separate time field)
      XML:  date = "MM-DD-YYYY", time = "H:MM" (legacy / fallback)

    Invalid events are logged and skipped. Emits a schema-drift warning
    if more than 5% of events fail validation.
    """
    provider = Provider.FOREX_FACTORY
    valid:         list[MFIPEvent] = []
    invalid_count: int             = 0
    now = datetime.now(timezone.utc)

    for raw in raw_events:
        try:
            title    = (raw.get("title") or "").strip()
            currency = (raw.get("country") or "").strip()   # FF "country" field = ISO currency

            # JSON uses a combined ISO date; XML uses separate date + time fields
            date_str  = raw.get("date", "")
            time_str  = raw.get("time", "")

            timestamp_utc = parse_event_datetime(date_str, time_str)
            is_all_day    = timestamp_utc is None and bool(date_str)
            impact        = normalize_impact(raw.get("impact", ""))
            # CDN feed is forward-looking; actual is absent but handle it if present
            actual        = normalize_value(raw.get("actual"))
            is_speech     = infer_is_speech(title)

            provider_eid = build_provider_event_id(title, currency, timestamp_utc)
            event_id     = build_event_id(provider, title, currency, timestamp_utc)

            event = MFIPEvent(
                event_id          = event_id,
                provider          = provider,
                provider_event_id = provider_eid,
                title             = title,
                currency          = currency,
                country           = currency_to_country(currency),
                timestamp_utc     = timestamp_utc,
                is_all_day        = is_all_day,
                impact            = impact,
                is_high_impact    = impact == ImpactLevel.HIGH,
                is_speech         = is_speech,
                category          = EventCategory.SPEECH if is_speech else infer_category(title),
                forecast          = normalize_value(raw.get("forecast")),
                previous          = normalize_value(raw.get("previous")),
                actual            = actual,
                status            = _derive_status(actual),
                last_updated      = now,
                metadata          = {
                    "source_week": source_week,
                    "raw_date":    date_str,
                    "raw_time":    time_str,
                    "raw_impact":  raw.get("impact", ""),
                    "url":         raw.get("url", ""),
                },
            )
            valid.append(event)

        except Exception as exc:
            invalid_count += 1
            logger.warning(f"Skipping invalid FF event ({raw.get('title', '?')}): {exc}")

    total = len(raw_events)
    if total > 0 and invalid_count / total > 0.05:
        logger.error(
            f"Schema drift: {invalid_count}/{total} FF events failed validation "
            f"({invalid_count/total:.0%}) — FF may have changed their JSON structure"
        )

    return valid
