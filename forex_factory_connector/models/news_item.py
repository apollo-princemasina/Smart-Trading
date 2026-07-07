# Reserved — Phase 3
# NewsItem model for forexfactory.com/news HTML scraping
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class NewsItem(BaseModel):
    news_id: str
    headline: str
    url: str
    published_at: Optional[datetime] = None
    currency_tags: list[str] = []
    fetched_at: datetime
