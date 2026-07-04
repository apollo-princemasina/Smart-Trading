"""Async Twelve Data API client.

Fetches OHLCV candles for EURUSD across all required timeframes.
Returns a standardised DataFrame ready for the RollingBufferManager.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx
import pandas as pd
from loguru import logger

from src.api.core.config import settings


class TwelveDataClient:
    """Thin async wrapper around the Twelve Data /time_series endpoint."""

    _TF_MAP: dict[str, str] = {
        "M15": "15min",
        "H1":  "1h",
        "H4":  "4h",
        "D1":  "1day",
        "W1":  "1week",
    }

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key  = api_key  or settings.TWELVE_DATA_API_KEY
        self._base_url = base_url or settings.TWELVE_DATA_BASE_URL

    # ── Public interface ──────────────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol:     str,
        timeframe:  str,
        outputsize: int,
    ) -> pd.DataFrame:
        """Download *outputsize* candles for *symbol* at *timeframe*.

        Returns a DataFrame sorted ascending (oldest → newest) with columns:
            timestamp, open, high, low, close, tick_volume, spread, real_volume
        """
        interval = self._TF_MAP.get(timeframe.upper())
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Valid: {list(self._TF_MAP)}")

        params = {
            "symbol":     symbol,
            "interval":   interval,
            "outputsize": outputsize,
            "apikey":     self._api_key,
            "format":     "JSON",
            "timezone":   "UTC",
        }

        logger.debug("Twelve Data fetch  symbol={} tf={} n={}", symbol, timeframe, outputsize)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self._base_url}/time_series", params=params)
            resp.raise_for_status()
            data = resp.json()

        if "values" not in data:
            msg = data.get("message", str(data))
            raise RuntimeError(f"Twelve Data error for {symbol}/{timeframe}: {msg}")

        df = self._parse(data["values"])
        logger.debug(
            "Twelve Data ok  symbol={} tf={}  rows={}  [{} → {}]",
            symbol, timeframe, len(df),
            df["timestamp"].iloc[0] if not df.empty else "?",
            df["timestamp"].iloc[-1] if not df.empty else "?",
        )
        return df

    async def fetch_all_timeframes(
        self,
        symbol:      str,
        buffer_sizes: dict[str, int],
        *,
        delay_s: float = 1.2,
    ) -> dict[str, pd.DataFrame]:
        """Fetch all 5 timeframes sequentially (respects free-tier rate limits).

        Parameters
        ----------
        symbol : str
        buffer_sizes : dict  e.g. {"M15": 800, "H1": 500, ...}
        delay_s : float
            Seconds to wait between requests (Twelve Data free tier ~8 req/min).
        """
        results: dict[str, pd.DataFrame] = {}
        for i, (tf, size) in enumerate(buffer_sizes.items()):
            try:
                df = await self.fetch_candles(symbol, tf, size)
                results[tf] = df
                logger.info("Buffer populated  tf={}  rows={}", tf, len(df))
            except Exception as exc:
                logger.error("Failed to fetch {}  {}: {}", symbol, tf, exc)
                results[tf] = pd.DataFrame()

            # Rate-limit delay between requests (skip after last)
            if i < len(buffer_sizes) - 1:
                await asyncio.sleep(delay_s)

        return results

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(values: list[dict]) -> pd.DataFrame:
        """Convert raw API values list into a clean OHLCV DataFrame."""
        df = pd.DataFrame(values)
        df = df.rename(columns={"datetime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        if "volume" in df.columns:
            df["tick_volume"] = (
                pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            )
        else:
            df["tick_volume"] = 0

        # Inject standard fields missing from Twelve Data FX responses
        df["spread"]      = settings.INFERENCE_SPREAD_FILL   # 1.5 pips
        df["real_volume"] = 0

        # Sort oldest → newest
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Keep only required columns in order
        keep = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        return df[[c for c in keep if c in df.columns]]
