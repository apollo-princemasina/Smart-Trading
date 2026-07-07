"""
Economic value parser — converts raw string values from MFIPEvent fields to floats.

Handles:
  "185K"     → 185_000.0
  "0.2%"     → 0.2
  "-1.2B"    → -1_200_000_000.0
  "1,234.5"  → 1_234.5
  "2.25"     → 2.25
  "−0.2%"    → -0.2   (unicode minus)
  "+0.1%"    → 0.1
  "N/A", ""  → None
"""
from __future__ import annotations

import re
from typing import Optional

# Multiplier suffixes (case-insensitive)
_MULTIPLIERS: dict[str, float] = {
    "K": 1e3,
    "M": 1e6,
    "B": 1e9,
    "T": 1e12,
}

# Values that represent "no data"
_NONE_SENTINELS = frozenset({
    "", "n/a", "na", "none", "null", "pending", "-", "–", "—", "tba", "tbd",
})

# Regex: optional sign, digits/commas/dots, optional suffix
_VALUE_RE = re.compile(
    r"^([+\-−]?)\s*([0-9,]+(?:\.[0-9]*)?)([KMBT]?)(%?)$",
    re.IGNORECASE,
)


def parse_economic_value(raw: Optional[str]) -> Optional[float]:
    """
    Parse a raw economic value string to float.

    Returns None when the value is absent, non-numeric, or a sentinel.
    Percentage signs are stripped — the returned float is the plain number
    (so "0.2%" → 0.2, not 0.002).
    """
    if raw is None:
        return None

    cleaned = raw.strip().replace("−", "-")  # unicode minus → ASCII minus

    if cleaned.lower() in _NONE_SENTINELS:
        return None

    # Remove thousand-separator commas before matching
    cleaned = cleaned.replace(",", "")

    m = _VALUE_RE.match(cleaned)
    if not m:
        return None

    sign_str, number_str, suffix, _pct = m.groups()

    try:
        value = float(number_str)
    except ValueError:
        return None

    if suffix.upper() in _MULTIPLIERS:
        value *= _MULTIPLIERS[suffix.upper()]

    if sign_str in ("-", "−"):
        value = -value

    return value
