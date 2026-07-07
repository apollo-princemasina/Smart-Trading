import httpx
from typing import Optional
from ..utils.config import settings
from ..utils.logger import logger


class HTTPClient:
    """Shared async HTTP client with ETag support and consistent headers."""

    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get(cls) -> httpx.AsyncClient:
        if cls._client is None or cls._client.is_closed:
            cls._client = httpx.AsyncClient(
                timeout=settings.REQUEST_TIMEOUT_S,
                headers={
                    "User-Agent":      settings.USER_AGENT,
                    "Accept":          "application/json, */*",
                    "Accept-Encoding": "gzip, deflate, br",
                },
                follow_redirects=True,
            )
        return cls._client

    @classmethod
    async def close(cls) -> None:
        if cls._client and not cls._client.is_closed:
            await cls._client.aclose()
            logger.info("HTTP client closed")
