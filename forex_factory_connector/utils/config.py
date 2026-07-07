from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parents[2]


class ConnectorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FF_", env_file=".env", extra="ignore")

    # Disk cache directory for calendar JSON and ETags — survives restarts
    DISK_CACHE_DIR: str = str(_BASE_DIR / "data" / "ff_cache")
    # Disk cache TTL in hours — serve from disk if CDN is unavailable
    DISK_CACHE_TTL_HOURS: float = 6.0

    # CDN base URL — configurable so a URL change doesn't require a code change
    # Confirmed by live research: hostname is nfs.faireconomy.media (no cdn- prefix)
    CDN_BASE_URL: str = "https://nfs.faireconomy.media"

    # Poll intervals (seconds)
    CALENDAR_POLL_SECONDS: int = 300        # 5 min — matches CDN max-age
    NEWS_POLL_SECONDS: int = 120            # 2 min — reserved Phase 3
    SENTIMENT_POLL_SECONDS: int = 300       # 5 min — reserved Phase 3
    SPEECHES_POLL_SECONDS: int = 600        # 10 min — derived from calendar

    # Adaptive poll interval when a High-impact event is within this window
    HIGH_IMPACT_LOOKAHEAD_MINUTES: int = 15
    HIGH_IMPACT_POLL_SECONDS: int = 60

    # HTTP client
    REQUEST_TIMEOUT_S: float = 10.0
    MAX_RETRIES: int = 3
    INITIAL_BACKOFF_S: float = 5.0
    BACKOFF_MULTIPLIER: float = 2.0
    MAX_BACKOFF_S: float = 60.0

    # Circuit breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 5     # failures before opening
    CIRCUIT_RESET_S: int = 300             # half-open probe interval

    # User-Agent string identifies the connector to FF infrastructure
    USER_AGENT: str = "MFIP-FF-Connector/1.0 (private trading tool; prince.masina@agri-forge.net)"


settings = ConnectorSettings()
