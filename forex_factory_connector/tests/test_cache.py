import pytest
from datetime import datetime, timezone
from market_intel.models.event import MFIPEvent
from market_intel.models.enums import ImpactLevel, EventStatus, EventCategory
from ..cache.memory_cache import ConnectorCache, CacheNotReadyError


@pytest.fixture
def cache():
    return ConnectorCache()


@pytest.fixture
def sample_event():
    return MFIPEvent(
        event_id="abc123", provider="forex_factory", provider_event_id="def456",
        title="US CPI", currency="USD", country="US",
        timestamp_utc=datetime(2026, 7, 7, 12, 30, tzinfo=timezone.utc),
        is_all_day=False, impact=ImpactLevel.HIGH, is_high_impact=True,
        is_speech=False, category=EventCategory.INFLATION,
        status=EventStatus.SCHEDULED, last_updated=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_get_before_populate_raises(cache):
    with pytest.raises(CacheNotReadyError):
        await cache.get_calendar("thisweek")


@pytest.mark.asyncio
async def test_set_and_get(cache, sample_event):
    await cache.set_calendar("thisweek", [sample_event])
    result = await cache.get_calendar("thisweek")
    assert len(result.events) == 1
    assert isinstance(result.events[0], MFIPEvent)


@pytest.mark.asyncio
async def test_is_populated(cache, sample_event):
    assert cache.is_populated("thisweek") is False
    await cache.set_calendar("thisweek", [sample_event])
    assert cache.is_populated("thisweek") is True
