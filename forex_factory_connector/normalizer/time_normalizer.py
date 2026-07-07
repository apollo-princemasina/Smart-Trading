"""
Converts Forex Factory CDN timestamps to UTC datetime objects.

CONFIRMED by live research: the JSON endpoint at nfs.faireconomy.media returns
dates as ISO 8601 timestamps (already combined date+time), not separate
"Jul 07, 2026" / "8:30am" string pairs.

The timestamp still uses US Eastern Time (EST/EDT). Conversion to UTC is
required on every parse. DST is handled automatically via America/New_York.

XML vs JSON difference:
  XML:  date="07-07-2026"  time="8:30"  (separate, EST)
  JSON: date="2026-07-07T08:30:00-04:00"  (ISO 8601, already timezone-aware)
        OR: date="2026-07-07T08:30:00"    (ISO 8601, naive — assume EST)

Both paths normalise to a UTC-aware datetime or None.
"""
from datetime import datetime, timezone
from typing import Optional
from dateutil import parser as dateutil_parser
from dateutil.tz import gettz

_EASTERN = gettz("America/New_York")


def parse_event_datetime(date_str: str, time_str: str = "") -> Optional[datetime]:
    """
    Parse a FF date (+ optional time) into a UTC-aware datetime.

    Handles three input shapes:
      1. ISO 8601 with offset:   "2026-07-07T08:30:00-04:00" → UTC directly
      2. ISO 8601 naive:         "2026-07-07T08:30:00"       → assume Eastern, convert
      3. Separate date + time:   ("Jul 07, 2026", "8:30am")  → parse Eastern, convert
      4. All-day / empty:        ("Jul 07, 2026", "All Day") → return None

    Returns None for all-day events and unparseable inputs.
    """
    if not date_str:
        return None

    if time_str and time_str.strip().lower() in ("all day", "tentative", ""):
        return None

    # Try combined ISO 8601 first (the JSON format confirmed by research)
    raw = date_str.strip()
    if "T" in raw:
        try:
            dt = dateutil_parser.isoparse(raw)
            if dt.tzinfo is None:
                # Naive ISO — assume Eastern
                dt = dt.replace(tzinfo=_EASTERN)
            return dt.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            pass

    # Fall back to separate date + time strings (XML format / legacy)
    combined = f"{raw} {time_str.strip()}" if time_str else raw
    try:
        naive = dateutil_parser.parse(combined)
        eastern = naive.replace(tzinfo=_EASTERN)
        return eastern.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None
