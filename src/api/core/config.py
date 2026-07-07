"""Application configuration.

Single source of truth for every environment variable the MFIP application
reads.  All values come from the .env file — nothing is hard-coded here.

Usage
-----
    from src.api.core.config import settings
    print(settings.TWELVE_DATA_API_KEY)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels above src/api/core/
_BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME:        str  = "MFIP"
    APP_VERSION:     str  = "1.0.0"
    APP_ENV:         Literal["development", "staging", "production"] = "development"
    APP_DEBUG:       bool = False
    APP_HOST:        str  = "0.0.0.0"
    APP_PORT:        int  = 8000
    LOG_LEVEL:       str  = "INFO"

    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY:      str  = "change-me-in-production-use-openssl-rand-hex-32"
    ALLOWED_ORIGINS: str  = "http://localhost:3000"   # comma-separated

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default=f"sqlite+aiosqlite:///{_BASE_DIR / 'data' / 'mfip.db'}",
        description="Async SQLAlchemy URL. Use postgresql+asyncpg://... for production.",
    )

    # ── Twelve Data ───────────────────────────────────────────────────────────
    TWELVE_DATA_API_KEY: str = "d0cd8527748f45f0b0ee8f02791feedb"
    TWELVE_DATA_BASE_URL: str = "https://api.twelvedata.com"

    # ── Rolling Buffer ────────────────────────────────────────────────────────
    # Candle counts chosen to give the feature pipeline enough warmup bars.
    # 800 M15 bars = ~5.5 trading days   (200-bar rolling windows need this)
    # 500 H1  bars = ~21 trading days
    # 300 H4  bars = ~50 trading days
    # 200 D1  bars = ~40 calendar weeks
    # 100 W1  bars = ~2 years
    BUFFER_M15_SIZE: int = 800
    BUFFER_H1_SIZE:  int = 500
    BUFFER_H4_SIZE:  int = 300
    BUFFER_D1_SIZE:  int = 200
    BUFFER_W1_SIZE:  int = 100

    # Restart cache — transient parquet files, NOT a database.
    # If the backend restarts within BUFFER_CACHE_TTL_HOURS it loads from
    # this cache instead of re-fetching from Twelve Data.
    BUFFER_CACHE_ENABLED:   bool  = True
    BUFFER_CACHE_TTL_HOURS: float = 4.0
    BUFFER_CACHE_DIR:       str   = str(_BASE_DIR / "data" / "buffer_cache")

    # ── Model Bundle ──────────────────────────────────────────────────────────
    MODEL_BUNDLE_DIR: str = str(_BASE_DIR / "models" / "best_model")
    MODEL_SYMBOL:        str = "EURUSD"    # internal model name (no slash)
    TWELVE_DATA_SYMBOL:  str = "EUR/USD"  # Twelve Data API format requires slash

    # ── Inference ─────────────────────────────────────────────────────────────
    INFERENCE_MIN_CONFIDENCE:    float = 0.60
    INFERENCE_SPREAD_FILL:       int   = 15      # 1.5 pips (Twelve Data omits FX spread)
    INFERENCE_TP_ATR_MULT:       float = 3.0
    INFERENCE_SL_ATR_MULT:       float = 1.5
    # Strategy B gate: require all 3 models to agree (HIGH_CONVICTION) before
    # emitting a directional signal.  SETUP_FORMING is surfaced as an alert field
    # but direction stays HOLD so no trade fires.
    INFERENCE_REQUIRE_CONVICTION: bool  = True

    # ── Scheduler ────────────────────────────────────────────────────────────
    # Cron expression for M15 bar close: fire at :01 past every 15-min mark
    # (1-minute delay gives the candle time to finalise on Twelve Data's side)
    SCHEDULER_M15_CRON: str = "1,16,31,46 * * * *"

    # ── Deriv WebSocket API ───────────────────────────────────────────────────
    # App ID 1 = public test app (fine for single-user dev/staging).
    # For production with multiple users: register an app at developers.deriv.com
    # (requires Admin-scope token), then set DERIV_APP_ID=<your_id> in .env.
    DERIV_APP_ID: int = 1

    # ── MT5 (Windows local dev only — not used on Railway) ───────────────────
    MT5_LOGIN:    int = 0
    MT5_PASSWORD: str = ""
    MT5_SERVER:   str = ""

    # ── Auth (Phase 6 — foundation) ───────────────────────────────────────────
    # SECRET_KEY (above) is reused for JWT signing.
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int  = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS:   int  = 7
    AUTH_ENABLED:                    bool = False

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def base_dir(self) -> Path:
        return _BASE_DIR

    @property
    def model_bundle_path(self) -> Path:
        return Path(self.MODEL_BUNDLE_DIR)

    @property
    def buffer_cache_path(self) -> Path:
        p = Path(self.BUFFER_CACHE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def buffer_sizes(self) -> dict[str, int]:
        return {
            "M15": self.BUFFER_M15_SIZE,
            "H1":  self.BUFFER_H1_SIZE,
            "H4":  self.BUFFER_H4_SIZE,
            "D1":  self.BUFFER_D1_SIZE,
            "W1":  self.BUFFER_W1_SIZE,
        }

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalise_db_url(cls, v: str) -> str:
        """Auto-upgrade sync driver prefixes to async equivalents.

        Handles:
          - Railway injecting bare postgresql:// URLs
          - Legacy .env files with sqlite:// (no +aiosqlite)
        """
        v = str(v)
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        if v.startswith("sqlite://") and "+aiosqlite" not in v:
            return v.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.  Use this everywhere."""
    return Settings()


# Module-level alias — import as `from src.api.core.config import settings`
settings = get_settings()
