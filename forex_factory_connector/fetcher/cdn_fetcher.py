from typing import Optional
import httpx
from ..utils.config import settings
from ..utils.logger import logger
from ..utils.retry import async_retry
from .http_client import HTTPClient
from .rate_limiter import RateLimiter


_limiter = RateLimiter(calls_per_minute=20)  # very conservative for a 5-min poll


class CDNFetchResult:
    def __init__(
        self,
        body: Optional[bytes],
        etag: Optional[str],
        not_modified: bool,
        rate_limited: bool = False,
        not_found: bool = False,
    ):
        self.body = body
        self.etag = etag
        self.not_modified = not_modified
        self.rate_limited = rate_limited
        self.not_found = not_found   # 404 — week not published on CDN yet


@async_retry(exceptions=(httpx.TransportError, httpx.TimeoutException))
async def fetch_calendar(week: str, current_etag: Optional[str] = None) -> CDNFetchResult:
    """
    Fetch a weekly calendar JSON from the faireconomy.media CDN.

    week: "thisweek" | "nextweek" | "lastweek"

    Returns CDNFetchResult. If not_modified is True, body is None and the
    caller must serve the existing cache unchanged. If rate_limited is True,
    the caller should serve from disk cache and skip updating etag_store.
    """
    await _limiter.acquire()

    url = f"{settings.CDN_BASE_URL}/ff_calendar_{week}.json"
    headers = {}
    if current_etag:
        headers["If-None-Match"] = current_etag

    client = HTTPClient.get()
    response = await client.get(url, headers=headers)

    if response.status_code == 304:
        logger.debug(f"CDN 304 Not Modified — {week} cache still valid")
        return CDNFetchResult(body=None, etag=current_etag, not_modified=True)

    if response.status_code == 404:
        # CDN only publishes thisweek; lastweek/nextweek return 404 when unavailable.
        # Treat as graceful "not published yet" — not a failure.
        logger.debug(f"CDN 404 — {week} not published on CDN (normal for lastweek/nextweek)")
        return CDNFetchResult(body=None, etag=None, not_modified=False, not_found=True)

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        logger.warning(
            f"CDN 429 Too Many Requests — {week} (Retry-After: {retry_after}). "
            "Will serve from disk cache."
        )
        return CDNFetchResult(body=None, etag=current_etag, not_modified=False, rate_limited=True)

    response.raise_for_status()

    new_etag = response.headers.get("ETag")
    logger.info(f"CDN fetch OK — {week} ({len(response.content)} bytes, etag={new_etag})")
    return CDNFetchResult(body=response.content, etag=new_etag, not_modified=False)
