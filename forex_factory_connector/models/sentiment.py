# Reserved — Phase 3
# SentimentData model for forexfactory.com/sentiment HTML scraping
from datetime import datetime
from pydantic import BaseModel


class PairSentiment(BaseModel):
    pair: str       # e.g. "EURUSD"
    long_pct: float
    short_pct: float


class SentimentSnapshot(BaseModel):
    pairs: list[PairSentiment]
    fetched_at: datetime
    is_stale: bool = False
