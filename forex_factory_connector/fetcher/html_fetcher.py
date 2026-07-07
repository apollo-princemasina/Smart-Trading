# Reserved — Phase 3
#
# HTMLFetcher wraps requests to forexfactory.com main pages (news, sentiment).
# The main site is behind Cloudflare Bot Management. This module is a placeholder;
# Phase 3 will evaluate playwright-stealth or a proxy service before implementing.
#
# Do NOT implement until a Cloudflare bypass strategy has been formally decided.

from typing import NoReturn


async def fetch_news_html() -> NoReturn:
    raise NotImplementedError("Phase 3: news scraping not yet implemented")


async def fetch_sentiment_html() -> NoReturn:
    raise NotImplementedError("Phase 3: sentiment scraping not yet implemented")
