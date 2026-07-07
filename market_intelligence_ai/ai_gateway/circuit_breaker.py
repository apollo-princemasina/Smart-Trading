"""
CircuitBreaker — prevents cascading failures when the AI provider is unhealthy.

States:
  CLOSED   → normal operation (requests go through)
  OPEN     → provider failing — requests blocked, fallback returned
  HALF_OPEN → cooldown expired — one probe request allowed

Transitions:
  CLOSED  → OPEN       after N consecutive failures
  OPEN    → HALF_OPEN  after reset_timeout_s
  HALF_OPEN → CLOSED   on probe success
  HALF_OPEN → OPEN     on probe failure (reset timer)
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from market_intelligence_ai.utils.logger import logger


class CircuitBreaker:
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, threshold: int = 5, reset_timeout_s: int = 60) -> None:
        self._threshold        = threshold
        self._reset_timeout_s  = reset_timeout_s
        self._state            = self.CLOSED
        self._consecutive_fails: int = 0
        self._opened_at: Optional[datetime] = None

    def record_success(self) -> None:
        self._consecutive_fails = 0
        if self._state == self.HALF_OPEN:
            logger.info("CircuitBreaker → CLOSED (probe succeeded)")
        self._state = self.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_fails += 1
        if self._state == self.HALF_OPEN:
            logger.warning("CircuitBreaker probe failed → OPEN (reset timer)")
            self._opened_at = datetime.now(timezone.utc)
            self._state = self.OPEN
        elif self._consecutive_fails >= self._threshold and self._state == self.CLOSED:
            logger.error(
                "CircuitBreaker → OPEN ({} consecutive failures)", self._consecutive_fails
            )
            self._state = self.OPEN
            self._opened_at = datetime.now(timezone.utc)

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            # Check if cooldown has expired
            if self._opened_at and (
                datetime.now(timezone.utc) - self._opened_at
                > timedelta(seconds=self._reset_timeout_s)
            ):
                self._state = self.HALF_OPEN
                logger.info("CircuitBreaker → HALF_OPEN (cooldown expired)")
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == self.OPEN

    @property
    def allows_request(self) -> bool:
        s = self.state
        return s in (self.CLOSED, self.HALF_OPEN)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_fails
