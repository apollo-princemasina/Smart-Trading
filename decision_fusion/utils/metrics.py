"""DFE performance metrics — rolling window, lock-protected."""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional


class DFEMetrics:
    """Tracks Decision Fusion Engine performance over a rolling window."""

    def __init__(self, maxlen: int = 200) -> None:
        self._lock            = asyncio.Lock()
        self._processing_ms:  deque[float] = deque(maxlen=maxlen)
        self._total_decisions: int = 0
        self._wait_decisions:  int = 0
        self._buy_decisions:   int = 0
        self._sell_decisions:  int = 0

    async def record(
        self,
        processing_ms: float,
        recommendation: str,
    ) -> None:
        async with self._lock:
            self._processing_ms.append(processing_ms)
            self._total_decisions += 1
            rec = recommendation.upper()
            if rec == "BUY":
                self._buy_decisions += 1
            elif rec == "SELL":
                self._sell_decisions += 1
            else:
                self._wait_decisions += 1

    def snapshot(self) -> dict:
        times = list(self._processing_ms)
        avg_ms = sum(times) / len(times) if times else None
        return {
            "total_decisions":  self._total_decisions,
            "buy_decisions":    self._buy_decisions,
            "sell_decisions":   self._sell_decisions,
            "wait_decisions":   self._wait_decisions,
            "avg_processing_ms": round(avg_ms, 2) if avg_ms is not None else None,
            "min_processing_ms": round(min(times), 2) if times else None,
            "max_processing_ms": round(max(times), 2) if times else None,
        }
