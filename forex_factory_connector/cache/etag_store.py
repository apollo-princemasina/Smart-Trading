import json
from pathlib import Path
from typing import Optional

from ..utils.config import settings
from ..utils.logger import logger


class ETagStore:
    """
    Per-endpoint ETag storage for conditional HTTP GET requests.

    ETags are persisted to disk so they survive process restarts.
    On restart the CDN returns 304 Not Modified (no body, no rate-limit hit)
    instead of a full re-fetch that triggers 429.
    """

    def __init__(self) -> None:
        self._etags: dict[str, Optional[str]] = {}
        self._path = Path(settings.DISK_CACHE_DIR) / "etags.json"
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                self._etags = json.loads(self._path.read_text(encoding="utf-8"))
                logger.debug("ETagStore: loaded {} etags from disk", len(self._etags))
        except Exception as exc:
            logger.warning("ETagStore: could not load etags from disk — {}", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._etags), encoding="utf-8")
        except Exception as exc:
            logger.warning("ETagStore: could not save etags to disk — {}", exc)

    def get(self, key: str) -> Optional[str]:
        return self._etags.get(key)

    def set(self, key: str, etag: Optional[str]) -> None:
        self._etags[key] = etag
        self._save()

    def clear(self, key: str) -> None:
        self._etags.pop(key, None)
        self._save()


# Module-level singleton shared across scheduler jobs
etag_store = ETagStore()
