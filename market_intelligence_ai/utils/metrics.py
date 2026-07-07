"""
MIA metrics — rolling window tracking for gateway health monitoring.
Thread-safe via asyncio.Lock.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class GatewayMetrics:
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    total_requests:   int = 0
    cached_responses: int = 0
    provider_calls:   int = 0
    failed_requests:  int = 0
    retry_count:      int = 0

    _latencies:    deque = field(default_factory=lambda: deque(maxlen=200), repr=False)
    _tokens_in:    deque = field(default_factory=lambda: deque(maxlen=200), repr=False)
    _tokens_out:   deque = field(default_factory=lambda: deque(maxlen=200), repr=False)

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    async def record_cache_hit(self) -> None:
        async with self._lock:
            self.total_requests   += 1
            self.cached_responses += 1

    async def record_provider_call(
        self,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        retries: int = 0,
    ) -> None:
        async with self._lock:
            self.total_requests += 1
            self.provider_calls += 1
            self.retry_count    += retries
            self._latencies.append(latency_ms)
            self._tokens_in.append(tokens_in)
            self._tokens_out.append(tokens_out)

    async def record_failure(self, retries: int = 0) -> None:
        async with self._lock:
            self.total_requests  += 1
            self.provider_calls  += 1
            self.failed_requests += 1
            self.retry_count     += retries

    @property
    def avg_latency_ms(self) -> Optional[float]:
        if not self._latencies:
            return None
        return round(sum(self._latencies) / len(self._latencies), 1)

    @property
    def avg_tokens_in(self) -> Optional[float]:
        if not self._tokens_in:
            return None
        return round(sum(self._tokens_in) / len(self._tokens_in), 0)

    @property
    def avg_tokens_out(self) -> Optional[float]:
        if not self._tokens_out:
            return None
        return round(sum(self._tokens_out) / len(self._tokens_out), 0)

    @property
    def cache_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.cached_responses / self.total_requests, 3)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.failed_requests / self.total_requests, 3)

    def snapshot(self) -> dict:
        return {
            "total_requests":   self.total_requests,
            "cached_responses": self.cached_responses,
            "provider_calls":   self.provider_calls,
            "failed_requests":  self.failed_requests,
            "retry_count":      self.retry_count,
            "avg_latency_ms":   self.avg_latency_ms,
            "avg_tokens_in":    self.avg_tokens_in,
            "avg_tokens_out":   self.avg_tokens_out,
            "cache_hit_rate":   self.cache_hit_rate,
            "error_rate":       self.error_rate,
        }
