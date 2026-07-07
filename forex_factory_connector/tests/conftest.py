import pytest
from datetime import datetime, timezone


@pytest.fixture
def sample_raw_event() -> dict:
    return {
        "title":    "US Non-Farm Employment Change",
        "country":  "USD",
        "date":     "Jul 04, 2026",
        "time":     "8:30am",
        "impact":   "High",
        "forecast": "185K",
        "previous": "177K",
        "actual":   "",
    }


@pytest.fixture
def sample_raw_event_released() -> dict:
    return {
        "title":    "US Non-Farm Employment Change",
        "country":  "USD",
        "date":     "Jul 04, 2026",
        "time":     "8:30am",
        "impact":   "High",
        "forecast": "185K",
        "previous": "177K",
        "actual":   "206K",
    }
