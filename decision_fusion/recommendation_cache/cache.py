"""
Decision Cache — maintains current, previous, and history of Decision Objects.

Lifecycle:
  - store(decision)  : saves a new decision, promotes current → previous
  - current          : the most recent unexpired decision
  - previous         : the decision before current
  - history          : last N decisions (oldest first)
  - is_expired       : True when current decision has passed its expires_at
  - invalidate()     : force-expire the current decision (call when new intel arrives)

Thread-safe: all mutations are protected by an asyncio.Lock.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from decision_fusion.schema.decision_object import DecisionObject
from decision_fusion.utils.config import dfe_config
from decision_fusion.utils.logger import logger


class DecisionCache:
    """
    Maintains the lifecycle of Decision Objects produced by the DFE.

    Provides:
      - Single source of truth for the current active decision
      - Previous decision (for change detection)
      - Rolling history (maxlen = DFE_HISTORY_MAX_SIZE)
      - Expiry tracking
    """

    def __init__(self) -> None:
        self._lock:     asyncio.Lock             = asyncio.Lock()
        self._current:  Optional[DecisionObject] = None
        self._previous: Optional[DecisionObject] = None
        self._history:  deque[DecisionObject]    = deque(maxlen=dfe_config.DFE_HISTORY_MAX_SIZE)

    # ── Mutations ─────────────────────────────────────────────────────────────

    async def store(self, decision: DecisionObject) -> None:
        """Store a new decision, promoting current → previous."""
        async with self._lock:
            if self._current is not None:
                self._previous = self._current
            self._current = decision
            self._history.append(decision)
        logger.debug(
            "Decision cached: {} {} conf={:.1f}",
            decision.recommendation,
            decision.recommendation_strength,
            decision.decision_confidence,
        )

    async def invalidate(self) -> None:
        """Force-expire the current decision (e.g. when new intel arrives)."""
        async with self._lock:
            self._current = None
        logger.debug("Current decision invalidated")

    # ── Reads (no lock needed for immutable Pydantic objects) ─────────────────

    @property
    def current(self) -> Optional[DecisionObject]:
        return self._current

    @property
    def previous(self) -> Optional[DecisionObject]:
        return self._previous

    def get_history(self, limit: int = 20) -> list[DecisionObject]:
        """Return the last `limit` decisions (newest first)."""
        items = list(self._history)
        return list(reversed(items))[:limit]

    def is_expired(self) -> bool:
        """True when the current decision's expires_at has passed."""
        if self._current is None:
            return True
        now = datetime.now(timezone.utc)
        return now >= self._current.expires_at

    def age_seconds(self) -> Optional[float]:
        """Seconds since the current decision was generated (None if no decision)."""
        if self._current is None:
            return None
        now = datetime.now(timezone.utc)
        return (now - self._current.generated_at).total_seconds()

    def seconds_until_expiry(self) -> Optional[float]:
        """Seconds until the current decision expires (None if no decision)."""
        if self._current is None:
            return None
        now = datetime.now(timezone.utc)
        delta = (self._current.expires_at - now).total_seconds()
        return max(0.0, delta)

    def size(self) -> int:
        return len(self._history)

    def stats(self) -> dict:
        current = self._current
        return {
            "has_current":          current is not None,
            "has_previous":         self._previous is not None,
            "history_size":         len(self._history),
            "is_expired":           self.is_expired(),
            "current_recommendation": current.recommendation if current else None,
            "age_seconds":          self.age_seconds(),
            "seconds_until_expiry": self.seconds_until_expiry(),
        }


# Module-level singleton — used by the engine and API endpoints
decision_cache = DecisionCache()
