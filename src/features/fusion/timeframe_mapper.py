"""Timeframe string → metadata mappings and hierarchy utilities."""

from __future__ import annotations

import pandas as pd


class TimeframeMapper:
    """
    Maps timeframe strings to pandas Timedelta objects and metadata.

    Supported canonical names: ``"W"``, ``"D"``, ``"H4"``, ``"H1"``, ``"M15"``.
    """

    # ── Hierarchy (index 0 = highest, 4 = lowest) ─────────────────────────────
    HIERARCHY: list[str] = ["W", "D", "H4", "H1", "M15"]

    # ── Canonical → pandas Timedelta ─────────────────────────────────────────
    _TIMEDELTA: dict[str, pd.Timedelta] = {
        "W":   pd.Timedelta(weeks=1),
        "D":   pd.Timedelta(days=1),
        "H4":  pd.Timedelta(hours=4),
        "H1":  pd.Timedelta(hours=1),
        "M15": pd.Timedelta(minutes=15),
    }

    # ── Canonical → column prefix ─────────────────────────────────────────────
    _PREFIX: dict[str, str] = {
        "W":   "weekly",
        "D":   "daily",
        "H4":  "h4",
        "H1":  "h1",
        "M15": "m15",
    }

    # ── Canonical → minutes ───────────────────────────────────────────────────
    _MINUTES: dict[str, int] = {
        "W":   7 * 24 * 60,
        "D":   24 * 60,
        "H4":  4 * 60,
        "H1":  60,
        "M15": 15,
    }

    # ── Alias → canonical ────────────────────────────────────────────────────
    _ALIASES: dict[str, str] = {
        "1W": "W",  "WEEKLY": "W",  "WEEK": "W",
        "1D": "D",  "DAILY":  "D",  "DAY":  "D",
        "4H": "H4", "240M":   "H4",
        "1H": "H1", "60M":    "H1",
        "15M": "M15", "15MIN": "M15", "M15MIN": "M15",
    }

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def normalise(cls, tf: str) -> str:
        """Return canonical timeframe string (e.g. ``"4H"`` → ``"H4"``)."""
        upper = tf.upper().strip()
        return cls._ALIASES.get(upper, upper)

    @classmethod
    def validate(cls, tf: str) -> str:
        """Normalise and raise ``ValueError`` if the timeframe is unknown."""
        canon = cls.normalise(tf)
        if canon not in cls._TIMEDELTA:
            raise ValueError(
                f"Unknown timeframe {tf!r}. "
                f"Valid canonical names: {list(cls._TIMEDELTA)}"
            )
        return canon

    @classmethod
    def prefix(cls, tf: str) -> str:
        """Column prefix for *tf* (e.g. ``"H1"`` → ``"h1"``)."""
        return cls._PREFIX[cls.validate(tf)]

    @classmethod
    def timedelta(cls, tf: str) -> pd.Timedelta:
        """Duration of one bar in the given timeframe."""
        return cls._TIMEDELTA[cls.validate(tf)]

    @classmethod
    def minutes(cls, tf: str) -> int:
        """Duration in minutes (e.g. ``"H4"`` → 240)."""
        return cls._MINUTES[cls.validate(tf)]

    @classmethod
    def rank(cls, tf: str) -> int:
        """
        Position in the hierarchy.  0 = highest (Weekly), 4 = lowest (M15).
        Lower rank → broader timeframe.
        """
        return cls.HIERARCHY.index(cls.validate(tf))

    @classmethod
    def is_higher_than(cls, tf_a: str, tf_b: str) -> bool:
        """True when *tf_a* covers a broader period than *tf_b*."""
        return cls.rank(tf_a) < cls.rank(tf_b)

    @classmethod
    def higher_timeframes(cls, base: str) -> list[str]:
        """All timeframes higher than *base*, ordered high → low."""
        base_rank = cls.rank(base)
        return [tf for tf in cls.HIERARCHY if cls.rank(tf) < base_rank]

    @classmethod
    def lower_timeframes(cls, base: str) -> list[str]:
        """All timeframes lower than *base*, ordered high → low."""
        base_rank = cls.rank(base)
        return [tf for tf in cls.HIERARCHY if cls.rank(tf) > base_rank]

    @classmethod
    def is_valid(cls, tf: str) -> bool:
        """True if *tf* (including aliases) is a recognised timeframe."""
        return cls.normalise(tf) in cls._TIMEDELTA
