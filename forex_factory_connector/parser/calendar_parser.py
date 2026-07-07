"""
Raw JSON deserializer for the nfs.faireconomy.media CDN calendar feed.

Confirmed JSON shape (from live research):
  {
    "title":    "US Non-Farm Employment Change",
    "country":  "USD",
    "date":     "2026-07-04T12:30:00-04:00",   ← ISO 8601 with Eastern offset
    "impact":   "High",                         ← "High" | "Medium" | "Low" | "Holiday"
    "forecast": "185K",
    "previous": "177K"
    // Note: no "actual" field in the CDN JSON — it is a forward-looking feed.
    // "actual" may be added by FF in the future or sourced from a different endpoint.
  }

The time field ("8:30am") appears in the XML format but NOT the JSON format.
The parser accepts both shapes and normalises them into a consistent raw dict.
"""
import json
from typing import Any


def parse_calendar_json(raw: bytes) -> list[dict[str, Any]]:
    """
    Deserialise raw CDN response bytes into a list of raw event dicts.

    Does no validation, transformation, or timezone conversion — those are
    the validator's and normalizer's responsibility.

    Raises ValueError on non-JSON or non-array responses.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"CDN returned invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at root, got {type(data).__name__}")

    return data
