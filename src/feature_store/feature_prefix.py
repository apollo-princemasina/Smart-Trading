"""Valid feature-name prefix definitions and validation utilities."""

from __future__ import annotations

from .exceptions import InvalidPrefixError

# ── Canonical prefix registry ─────────────────────────────────────────────────

VALID_PREFIXES: frozenset[str] = frozenset(
    [
        # Timeframe-aligned HTF context
        "weekly_",
        "daily_",
        "h4_",
        "h1_",
        "m15_",
        # Domain engines
        "ms_",      # Market Structure (BOS, CHoCH, OB, FVG)
        "liq_",     # Liquidity (sweeps, equal H/L, pools)
        "tech_",    # Traditional Technical Indicators
        "stat_",    # Statistical & Microstructure
        "vol_",     # Volatility
        "sess_",    # Session markers (London, NY, Asia)
        # Output / target columns
        "label_",   # Label / target columns
        "future_",  # Forward-looking values (label pipeline only)
        # Future data sources
        "macro_",   # Macroeconomic data
        "news_",    # News sentiment
        "sent_",    # Market sentiment / COT / options flow
    ]
)

PREFIX_DESCRIPTIONS: dict[str, str] = {
    "weekly_":  "Weekly timeframe features",
    "daily_":   "Daily timeframe features",
    "h4_":      "4-hour timeframe features",
    "h1_":      "1-hour timeframe features",
    "m15_":     "15-minute base-timeframe features",
    "ms_":      "Market Structure Engine (ICT / LuxAlgo SMC)",
    "liq_":     "Liquidity Engine (sweeps, pools, equal H/L)",
    "tech_":    "Traditional Technical Indicator Engine",
    "stat_":    "Statistical & Market Microstructure Engine",
    "vol_":     "Volatility Engine",
    "sess_":    "Session Engine (London / NY / Asia markers)",
    "label_":   "Label / target columns",
    "future_":  "Forward-looking values (label pipeline only)",
    "macro_":   "Macroeconomic data",
    "news_":    "News sentiment features",
    "sent_":    "Market sentiment, COT, options-flow features",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def extract_prefix(feature_name: str) -> str:
    """
    Extract the prefix portion from *feature_name*.

    Returns the longest matching valid prefix, or an empty string if
    no registered prefix matches.

    Examples
    --------
    >>> extract_prefix("h1_rsi")
    'h1_'
    >>> extract_prefix("weekly_bos")
    'weekly_'
    >>> extract_prefix("unknown_feature")
    ''
    """
    for prefix in sorted(VALID_PREFIXES, key=len, reverse=True):
        if feature_name.startswith(prefix):
            return prefix
    return ""


def validate_prefix(feature_name: str) -> str:
    """
    Validate that *feature_name* starts with a registered prefix.

    Returns the matched prefix string.

    Raises
    ------
    InvalidPrefixError
        If no valid prefix matches.
    """
    prefix = extract_prefix(feature_name)
    if not prefix:
        raise InvalidPrefixError(
            f"Feature '{feature_name}' does not start with any valid prefix. "
            f"Valid prefixes: {sorted(VALID_PREFIXES)}"
        )
    return prefix


def validate_all_prefixes(feature_names: list[str]) -> dict[str, str]:
    """
    Validate every name in *feature_names*.

    Returns a mapping of ``{feature_name: prefix}``.

    Raises
    ------
    InvalidPrefixError
        Listing all invalid feature names in one call.
    """
    result: dict[str, str] = {}
    invalid: list[str] = []

    for name in feature_names:
        prefix = extract_prefix(name)
        if prefix:
            result[name] = prefix
        else:
            invalid.append(name)

    if invalid:
        raise InvalidPrefixError(
            f"{len(invalid)} feature(s) have invalid prefixes: {invalid}. "
            f"Valid prefixes: {sorted(VALID_PREFIXES)}"
        )
    return result


def group_by_prefix(feature_names: list[str]) -> dict[str, list[str]]:
    """Return ``{prefix: [feature_names]}`` grouped by prefix."""
    groups: dict[str, list[str]] = {}
    for name in feature_names:
        pfx = extract_prefix(name) or "__unknown__"
        groups.setdefault(pfx, []).append(name)
    return groups


def is_valid_prefix(feature_name: str) -> bool:
    """Return True if *feature_name* starts with a registered prefix."""
    return bool(extract_prefix(feature_name))
