"""Tests for AICache — TTL behaviour, hit/miss tracking, eviction."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from market_intelligence_ai.ai_cache.cache import AICache


@pytest.fixture
def cache():
    return AICache()


@pytest.mark.asyncio
async def test_set_and_get(cache):
    await cache.set("k1", {"value": 42}, ttl_seconds=60)
    result = await cache.get("k1")
    assert result == {"value": 42}


@pytest.mark.asyncio
async def test_miss_returns_none(cache):
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_expired_entry_returns_none(cache):
    # Set with TTL -1 sec → already expired
    entry_data = "stale"
    await cache.set("expired_key", entry_data, ttl_seconds=1)

    # Manually expire by patching the entry
    async with cache._lock:
        cache._store["expired_key"].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    result = await cache.get("expired_key")
    assert result is None


@pytest.mark.asyncio
async def test_hit_rate_tracking(cache):
    await cache.set("k1", "v1", ttl_seconds=60)

    await cache.get("k1")           # hit
    await cache.get("k1")           # hit
    await cache.get("missing_key")  # miss

    assert cache.hit_rate == pytest.approx(2 / 3, abs=0.01)


@pytest.mark.asyncio
async def test_clear_resets_stats(cache):
    await cache.set("k1", "v1", ttl_seconds=60)
    await cache.get("k1")
    await cache.clear()

    assert cache.hit_rate == 0.0
    assert cache.total_entries == 0
    assert await cache.get("k1") is None


@pytest.mark.asyncio
async def test_evict_expired_removes_stale_entries(cache):
    await cache.set("fresh", "value", ttl_seconds=3600)
    await cache.set("stale", "value", ttl_seconds=1)

    # Manually expire the stale entry
    async with cache._lock:
        cache._store["stale"].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    evicted = await cache.evict_expired()
    assert evicted == 1
    assert cache.total_entries == 1
    assert await cache.get("fresh") is not None


@pytest.mark.asyncio
async def test_invalidate_removes_entry(cache):
    await cache.set("k1", "v1", ttl_seconds=60)
    await cache.invalidate("k1")
    assert await cache.get("k1") is None


@pytest.mark.asyncio
async def test_stats_structure(cache):
    stats = cache.stats()
    assert set(stats.keys()) == {"hits", "misses", "hit_rate", "total_entries"}


@pytest.mark.asyncio
async def test_concurrent_set_and_get(cache):
    """Verify no data races under concurrent access."""
    async def writer(i: int):
        await cache.set(f"key_{i}", i, ttl_seconds=60)

    async def reader(i: int):
        return await cache.get(f"key_{i}")

    await asyncio.gather(*[writer(i) for i in range(50)])
    results = await asyncio.gather(*[reader(i) for i in range(50)])
    assert all(r is not None for r in results)
