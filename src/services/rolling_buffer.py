"""Rolling Buffer Manager — single source of truth for live market data.

Architecture
------------
- One deque per timeframe, maxlen = configured buffer size.
- On startup: loads restart cache if fresh, otherwise downloads from MT5.
- On update (every M15 close): fetches last 2 candles per timeframe, appends
  the confirmed-closed one (index 0), oldest auto-drops from deque.
- On shutdown: saves buffer to transient parquet cache files.
- All downstream modules call as_dataframe() — never touch buffers directly.

No OHLCV data is ever written to PostgreSQL.
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.api.core.config import settings


class RollingBufferManager:

    _SYMBOL = "EURUSD"    # MT5 symbol format — no slash

    def __init__(self) -> None:
        self._sizes: dict[str, int] = settings.buffer_sizes
        self._buffers: dict[str, deque[dict]] = {
            tf: deque(maxlen=size) for tf, size in self._sizes.items()
        }
        self._ready:  dict[str, bool] = {tf: False for tf in self._sizes}
        self._cache_dir: Path = settings.buffer_cache_path

    # ── Startup ───────────────────────────────────────────────────────────────

    async def initialise(self) -> None:
        """Populate all buffers.  Tries cache first, falls back to MT5."""
        loaded = await self.load_cache()
        if loaded:
            logger.info("Rolling buffers restored from restart cache")
        else:
            logger.info("Downloading rolling buffers from Deriv...")
            await self._download_all()

    async def _download_all(self) -> None:
        from src.services.deriv_client import DerivClient
        client = DerivClient()
        frames = await client.fetch_all_timeframes(self._SYMBOL, self._sizes)
        for tf, df in frames.items():
            if df.empty:
                logger.warning("Buffer for {} is empty after Deriv download", tf)
                continue
            # Drop the last bar — Deriv includes the currently-open bar whose
            # OHLC is not final. The scheduler will add it once it closes.
            if len(df) > 1:
                df = df.iloc[:-1]
            for row in df.to_dict("records"):
                self._buffers[tf].append(row)
            self._ready[tf] = True
        logger.info(
            "Buffers populated (last open bar excluded): {}",
            {tf: len(buf) for tf, buf in self._buffers.items()},
        )

    # ── Per-cycle update (called by scheduler) ───────────────────────────────

    # Timeframes updated on every M15 tick vs. lazily (to stay within free-tier rate limits).
    # W1 bars close once per week — no need to poll them every 15 minutes.
    _TICK_UPDATE_TFS = ("M15", "H1", "H4", "D1")

    async def update(self) -> dict[str, bool]:
        """Fetch the latest closed candle for each timeframe and append it.

        Returns a dict of {tf: updated} indicating which buffers got a new row.
        W1 is skipped on every tick (changes weekly, not every 15 min).
        """
        from src.services.deriv_client import DerivClient
        client = DerivClient()
        updated: dict[str, bool] = {}

        for tf in self._TICK_UPDATE_TFS:
            try:
                # Fetch 2 candles sorted ascending: [confirmed-closed, currently-open]
                # iloc[-2] = second-to-last = the bar that just confirmed closed
                # iloc[-1] = last = the currently-open bar (discard — OHLC not final)
                df = await client.fetch_candles(self._SYMBOL, tf, outputsize=2)
                if len(df) < 2:
                    updated[tf] = False
                    continue

                confirmed = df.iloc[-2].to_dict()  # the just-closed bar
                existing_ts = (
                    self._buffers[tf][-1]["timestamp"]
                    if self._buffers[tf]
                    else None
                )

                new_ts = confirmed["timestamp"]
                if existing_ts is None or new_ts >= existing_ts:
                    # >= so we also replace a startup bar that was open and is now closed
                    if existing_ts is not None and new_ts == existing_ts:
                        # Same timestamp: replace last bar with final OHLC
                        self._buffers[tf][-1] = confirmed
                    else:
                        self._buffers[tf].append(confirmed)
                    self._ready[tf] = True
                    updated[tf] = True
                    logger.debug("Buffer updated  tf={}  ts={}", tf, new_ts)
                else:
                    updated[tf] = False

            except Exception as exc:
                logger.warning("Buffer update failed for {}: {}", tf, exc)
                updated[tf] = False

        return updated

    # ── Data access ───────────────────────────────────────────────────────────

    def is_ready(self, timeframe: str) -> bool:
        return self._ready.get(timeframe.upper(), False)

    @property
    def all_ready(self) -> bool:
        # M15 + at least one HTF is enough to run inference.
        # D1 and W1 are enrichment TFs — their absence degrades but does not block.
        core_tfs = ("M15", "H1", "H4")
        return all(self._ready.get(tf, False) for tf in core_tfs)

    def get_candles(self, timeframe: str, limit: int = 0) -> list[dict[str, Any]]:
        tf = timeframe.upper()
        candles = list(self._buffers.get(tf, deque()))
        if limit:
            return candles[-limit:]
        return candles

    def latest_close(self, timeframe: str = "M15") -> float | None:
        candles = self.get_candles(timeframe)
        return float(candles[-1]["close"]) if candles else None

    def as_dataframe(self, timeframe: str) -> pd.DataFrame:
        """Return the buffer contents as a clean, sorted DataFrame."""
        candles = self.get_candles(timeframe)
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True)

    # ── Cache persistence ─────────────────────────────────────────────────────

    async def save_cache(self) -> None:
        """Write each buffer to a parquet file in the cache directory."""
        if not settings.BUFFER_CACHE_ENABLED:
            return
        meta = {"saved_at": datetime.now(timezone.utc).isoformat()}
        (self._cache_dir / "meta.json").write_text(json.dumps(meta))
        for tf, buf in self._buffers.items():
            if not buf:
                continue
            df = self.as_dataframe(tf)
            df.to_parquet(self._cache_dir / f"{tf}.parquet", index=False)
        logger.info("Buffer cache saved to {}", self._cache_dir)

    async def load_cache(self) -> bool:
        """Load buffers from cache if the cache is younger than TTL.

        Returns True if cache was loaded, False if stale/missing.
        """
        if not settings.BUFFER_CACHE_ENABLED:
            return False
        meta_path = self._cache_dir / "meta.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
            saved_at = datetime.fromisoformat(meta["saved_at"])
            age = datetime.now(timezone.utc) - saved_at
            if age > timedelta(hours=settings.BUFFER_CACHE_TTL_HOURS):
                logger.info("Buffer cache is stale ({:.0f}h old) — re-downloading", age.total_seconds() / 3600)
                return False

            for tf in self._sizes:
                path = self._cache_dir / f"{tf}.parquet"
                if not path.exists():
                    continue
                df = pd.read_parquet(path)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                for row in df.to_dict("records"):
                    self._buffers[tf].append(row)
                self._ready[tf] = len(self._buffers[tf]) > 0
            return True
        except Exception as exc:
            logger.warning("Failed to load buffer cache: {}", exc)
            return False
