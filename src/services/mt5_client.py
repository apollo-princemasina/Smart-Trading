"""MT5 live data client — drop-in replacement for TwelveDataClient.

Wraps the blocking MetaTrader5 C-extension in a ThreadPoolExecutor so it
never blocks the FastAPI event loop.  The MT5 terminal must be running and
logged in; initialize() re-uses the existing session without re-logging.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from loguru import logger

from src.api.core.config import settings

# One dedicated thread — MT5 library is not thread-safe across threads
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5")

_TF_MAP_NAMES: dict[str, str] = {
    "M15": "TIMEFRAME_M15",
    "H1":  "TIMEFRAME_H1",
    "H4":  "TIMEFRAME_H4",
    "D1":  "TIMEFRAME_D1",
    "W1":  "TIMEFRAME_W1",
}


def _build_df(rates: Any) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop(columns=["time"])
    for col in ("spread", "real_volume"):
        if col not in df.columns:
            df[col] = 0
    keep = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    df = df[[c for c in keep if c in df.columns]]
    return df.sort_values("timestamp").reset_index(drop=True)


def _fetch_all_blocking(symbol: str, buffer_sizes: dict[str, int]) -> dict[str, pd.DataFrame]:
    """Fetch all timeframes in one MT5 session. Runs inside the thread executor."""
    import MetaTrader5 as mt5

    tf_map = {k: getattr(mt5, v) for k, v in _TF_MAP_NAMES.items()}

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    try:
        results: dict[str, pd.DataFrame] = {}
        for tf, count in buffer_sizes.items():
            tf_int = tf_map.get(tf)
            if tf_int is None:
                logger.warning("MT5: unknown timeframe {} — skipped", tf)
                continue
            rates = mt5.copy_rates_from_pos(symbol, tf_int, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning("MT5: no data for {} {}", symbol, tf)
                results[tf] = pd.DataFrame()
                continue
            results[tf] = _build_df(rates)
            logger.info("MT5: {} — {} bars loaded", tf, len(results[tf]))
        return results
    finally:
        mt5.shutdown()


def _fetch_latest_blocking(symbol: str, tf: str, count: int = 2) -> pd.DataFrame:
    """Fetch the last *count* bars for a single timeframe. Runs inside the thread executor."""
    import MetaTrader5 as mt5

    tf_map = {k: getattr(mt5, v) for k, v in _TF_MAP_NAMES.items()}
    tf_int = tf_map.get(tf)
    if tf_int is None:
        raise ValueError(f"Unknown timeframe: {tf}")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    try:
        rates = mt5.copy_rates_from_pos(symbol, tf_int, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5: no data for {symbol} {tf}")
        return _build_df(rates)
    finally:
        mt5.shutdown()


class MT5Client:
    """Async wrapper around the blocking MT5 API.

    Matches the interface of TwelveDataClient so RollingBufferManager
    needs minimal changes.
    """

    _SYMBOL = "EURUSD"  # MT5 format: no slash

    async def fetch_all_timeframes(
        self,
        symbol: str,
        buffer_sizes: dict[str, int],
    ) -> dict[str, pd.DataFrame]:
        """Fetch all timeframes in one MT5 session (no rate-limit delay needed)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            _fetch_all_blocking,
            symbol,
            buffer_sizes,
        )

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        outputsize: int,
    ) -> pd.DataFrame:
        """Fetch *outputsize* bars for a single timeframe."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            _fetch_latest_blocking,
            symbol,
            timeframe,
            outputsize,
        )
