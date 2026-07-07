"""
AICache — TTL-based in-memory cache for MarketIntelligenceOutput results.

Cache keys are deterministic and include context_schema_version so bumping
the version automatically invalidates all stale entries.

Key design:
  event key    = sha256(context_v1 | event | currency | event_id | surprise_class)[:32]
  headline key = sha256(context_v1 | headline | currency | headline_hash)[:32]
  combined key = sha256(context_v1 | combined | event_id | headline_hash)[:32]

Keys are always derived from ContextBuilder.cache_key(payload) — callers should
never construct cache keys manually.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from market_intelligence_ai.utils.logger import logger


@dataclass
class CacheEntry:
    data:       Any
    expires_at: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count:  int = 0

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def touch(self) -> None:
        self.hit_count += 1


class AICache:
    """
    Unified TTL cache for all MarketIntelligenceOutput results.

    Thread-safe via asyncio.Lock. Expired entries are evicted lazily on reads.
    """

    def __init__(self) -> None:
        self._lock    = asyncio.Lock()
        self._store:  dict[str, CacheEntry] = {}
        self._hits:   int = 0
        self._misses: int = 0

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.is_expired:
                if entry is not None:
                    del self._store[key]
                self._misses += 1
                return None
            entry.touch()
            self._hits += 1
            return entry.data

    async def set(self, key: str, data: Any, ttl_seconds: int) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        async with self._lock:
            self._store[key] = CacheEntry(data=data, expires_at=expires_at)
        logger.debug("AICache SET key={}… ttl={}s", key[:16], ttl_seconds)

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    async def evict_expired(self) -> int:
        async with self._lock:
            expired_keys = [k for k, v in self._store.items() if v.is_expired]
            for k in expired_keys:
                del self._store[k]
            return len(expired_keys)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / total, 3) if total > 0 else 0.0

    @property
    def total_entries(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "hits":          self._hits,
            "misses":        self._misses,
            "hit_rate":      self.hit_rate,
            "total_entries": self.total_entries,
        }


# Module-level singleton
ai_cache = AICache()
