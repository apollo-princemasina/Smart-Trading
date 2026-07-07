"""Deriv WebSocket API client — cross-platform replacement for MT5Client.

Uses the public Deriv candle API (wss://ws.binaryws.com/websockets/v3).
No authentication required for market data — works on Linux / Railway.

Matches the MT5Client interface: fetch_all_timeframes() and fetch_candles().
"""
from __future__ import annotations

import asyncio
import json

import pandas as pd
from loguru import logger

from src.api.core.config import settings

_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id={app_id}"

_SYMBOL_MAP: dict[str, str] = {
    "EURUSD": "frxEURUSD",
    "GBPUSD": "frxGBPUSD",
    "USDJPY": "frxUSDJPY",
    "XAUUSD": "frxXAUUSD",
}

# Deriv does not support W1 (604800) — handled via D1 resample below
_GRANULARITY: dict[str, int] = {
    "M1":  60,
    "M5":  300,
    "M15": 900,
    "H1":  3600,
    "H4":  14400,
    "D1":  86400,
}


async def _ws_fetch(deriv_symbol: str, granularity: int, count: int) -> list[dict]:
    """Single request over a fresh connection — use _ws_fetch_multi for bulk loads."""
    import websockets
    url = _WS_URL.format(app_id=settings.DERIV_APP_ID)
    req = {
        "ticks_history": deriv_symbol,
        "count": count,
        "end": "latest",
        "granularity": granularity,
        "style": "candles",
        "adjust_start_time": 1,
    }
    async with websockets.connect(url, open_timeout=15) as ws:
        await ws.send(json.dumps(req))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
    data = json.loads(raw)
    if "error" in data:
        raise RuntimeError(f"Deriv API error: {data['error']['message']}")
    return data.get("candles", [])


async def _ws_fetch_multi(
    deriv_symbol: str,
    requests: list[dict],  # list of {tf, granularity, count}
) -> dict[str, list[dict]]:
    """Fetch multiple timeframes over ONE WebSocket connection."""
    import websockets
    url = _WS_URL.format(app_id=settings.DERIV_APP_ID)

    results: dict[str, list[dict]] = {}
    async with websockets.connect(url, open_timeout=15) as ws:
        for req_meta in requests:
            tf = req_meta["tf"]
            req = {
                "ticks_history": deriv_symbol,
                "count": req_meta["count"],
                "end": "latest",
                "granularity": req_meta["granularity"],
                "style": "candles",
                "adjust_start_time": 1,
            }
            await ws.send(json.dumps(req))
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            data = json.loads(raw)
            if "error" in data:
                logger.warning("Deriv API error for {}: {}", tf, data["error"]["message"])
                results[tf] = []
            else:
                results[tf] = data.get("candles", [])
    return results


def _resample_w1(d1_df: pd.DataFrame) -> pd.DataFrame:
    """Resample a D1 DataFrame into W1 (Monday-anchored weekly) bars."""
    if d1_df.empty:
        return d1_df
    df = d1_df.set_index("timestamp")
    w1 = df.resample("W-MON", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        spread=("spread", "first"),
        real_volume=("real_volume", "sum"),
    ).dropna(subset=["open"])
    w1 = w1.reset_index().rename(columns={"timestamp": "timestamp"})
    return w1.sort_values("timestamp").reset_index(drop=True)


def _to_df(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.drop(columns=["epoch"])
    df["tick_volume"] = 0
    df["spread"] = settings.INFERENCE_SPREAD_FILL
    df["real_volume"] = 0
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    return df[[c for c in keep if c in df.columns]].sort_values("timestamp").reset_index(drop=True)


class DerivClient:
    """Async Deriv API client.  Drop-in replacement for MT5Client."""

    def _sym(self, symbol: str) -> str:
        return _SYMBOL_MAP.get(symbol.upper(), f"frx{symbol.upper()}")

    async def fetch_all_timeframes(
        self,
        symbol: str,
        buffer_sizes: dict[str, int],
    ) -> dict[str, pd.DataFrame]:
        deriv_sym = self._sym(symbol)

        # Build request list — W1 fetches as D1 for resampling
        req_list = []
        w1_count = 0
        for tf, count in buffer_sizes.items():
            if tf == "W1":
                w1_count = count
                req_list.append({"tf": "W1_raw", "granularity": 86400, "count": count * 7 + 14})
            else:
                gran = _GRANULARITY.get(tf)
                if gran is None:
                    logger.warning("DerivClient: unknown timeframe {} — skipped", tf)
                    continue
                req_list.append({"tf": tf, "granularity": gran, "count": count})

        # Single connection, all requests sequential over it
        try:
            raw_results = await _ws_fetch_multi(deriv_sym, req_list)
        except Exception as exc:
            logger.warning("Deriv: multi-fetch failed: {}", exc)
            return {tf: pd.DataFrame() for tf in buffer_sizes}

        out: dict[str, pd.DataFrame] = {}
        for tf in buffer_sizes:
            if tf == "W1":
                candles = raw_results.get("W1_raw", [])
                df = _resample_w1(_to_df(candles)).tail(w1_count).reset_index(drop=True)
                logger.info("Deriv: W1 — {} bars (resampled from D1)", len(df))
                out["W1"] = df
            else:
                candles = raw_results.get(tf, [])
                df = _to_df(candles)
                logger.info("Deriv: {} — {} bars loaded", tf, len(df))
                out[tf] = df
        return out

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        outputsize: int,
    ) -> pd.DataFrame:
        deriv_sym = self._sym(symbol)
        tf = timeframe.upper()
        if tf == "W1":
            d1_count = outputsize * 7 + 14
            candles = await _ws_fetch(deriv_sym, 86400, d1_count)
            return _resample_w1(_to_df(candles)).tail(outputsize).reset_index(drop=True)
        gran = _GRANULARITY.get(tf)
        if gran is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        candles = await _ws_fetch(deriv_sym, gran, outputsize)
        return _to_df(candles)
