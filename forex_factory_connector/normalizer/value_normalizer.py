"""
Converts raw FF value strings (forecast/previous/actual) to clean Optional[str].

FF uses empty string to mean "not available"; we normalise to None so downstream
code can use a simple None check rather than `val is not None and val != ""`.
"""
from typing import Optional


def normalize_value(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None
