"""Trend indicators: ADX/±DI, Aroon, Parabolic SAR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_ADX_PERIOD   = 14
_AROON_PERIOD = 25
_PSAR_START   = 0.02
_PSAR_STEP    = 0.02
_PSAR_MAX     = 0.20

_TREND_COLUMNS: list[str] = [
    "adx", "plus_di", "minus_di",
    "aroon_up", "aroon_down", "aroon_oscillator",
    "parabolic_sar",
]


def _wilder_rma(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).ewm(com=period - 1, adjust=False).mean().to_numpy()


def _parabolic_sar(high: np.ndarray, low: np.ndarray,
                   start: float, step: float, max_af: float) -> np.ndarray:
    """Wilder's Parabolic SAR — O(N) Python loop (no vectorised equivalent)."""
    n = len(high)
    sar = np.empty(n, dtype=np.float64)
    if n == 0:
        return sar

    sar[0]  = low[0]
    bull    = True
    ep      = high[0]
    af      = start

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, low[i - 1], low[max(0, i - 2)])
            if low[i] < new_sar:
                bull    = False
                new_sar = ep
                ep      = low[i]
                af      = start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_af)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, high[i - 1], high[max(0, i - 2)])
            if high[i] > new_sar:
                bull    = True
                new_sar = ep
                ep      = high[i]
                af      = start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_af)
        sar[i] = new_sar

    return sar


@FeatureRegistry.register
class TrendEngine(BaseFeature):
    """ADX/+DI/-DI (14), Aroon Up/Down/Oscillator (25), Parabolic SAR."""

    name:             str       = "trend"
    category:         str       = "technical"
    dependencies:     list[str] = []
    required_columns: list[str] = ["high", "low", "close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        high  = df["high"].to_numpy(dtype=np.float64)
        low   = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)

        prev_high  = np.concatenate([[high[0]],  high[:-1]])
        prev_low   = np.concatenate([[low[0]],   low[:-1]])
        prev_close = np.concatenate([[close[0]], close[:-1]])

        # ── True Range ────────────────────────────────────────────────────────
        tr = np.maximum(high - low,
             np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

        # ── Directional Movement ───────────────────────────────────────────────
        up_move   = high - prev_high
        down_move = prev_low - low
        plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # ── Wilder-smoothed TR, +DM, -DM ──────────────────────────────────────
        atr14     = _wilder_rma(tr, _ADX_PERIOD)
        plus_dm14 = _wilder_rma(plus_dm, _ADX_PERIOD)
        minus_dm14= _wilder_rma(minus_dm, _ADX_PERIOD)

        safe_atr = np.where(atr14 > 0, atr14, 1.0)
        plus_di  = np.where(atr14 > 0, plus_dm14  / safe_atr * 100.0, 0.0)
        minus_di = np.where(atr14 > 0, minus_dm14 / safe_atr * 100.0, 0.0)

        di_sum   = plus_di + minus_di
        di_diff  = np.abs(plus_di - minus_di)
        safe_sum = np.where(di_sum > 0, di_sum, 1.0)
        dx       = np.where(di_sum > 0, di_diff / safe_sum * 100.0, 0.0)
        adx      = _wilder_rma(dx, _ADX_PERIOD)

        # ── Aroon (25-bar period) ──────────────────────────────────────────────
        high_s = pd.Series(high, index=df.index)
        low_s  = pd.Series(low,  index=df.index)

        win = _AROON_PERIOD + 1
        aroon_up   = high_s.rolling(win, min_periods=1).apply(
            lambda x: (len(x) - 1 - np.argmax(x[::-1])) / _AROON_PERIOD * 100.0,
            raw=True).to_numpy()
        aroon_down = low_s.rolling(win, min_periods=1).apply(
            lambda x: (len(x) - 1 - np.argmin(x[::-1])) / _AROON_PERIOD * 100.0,
            raw=True).to_numpy()
        aroon_oscillator = aroon_up - aroon_down

        # ── Parabolic SAR ──────────────────────────────────────────────────────
        psar = _parabolic_sar(high, low, _PSAR_START, _PSAR_STEP, _PSAR_MAX)

        out = pd.DataFrame(index=df.index)
        out["adx"]              = adx
        out["plus_di"]          = plus_di
        out["minus_di"]         = minus_di
        out["aroon_up"]         = aroon_up
        out["aroon_down"]       = aroon_down
        out["aroon_oscillator"] = aroon_oscillator
        out["parabolic_sar"]    = psar
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "ADX-14, +DI-14, -DI-14 (Wilder's directional system); "
                "Aroon Up/Down/Oscillator-25; Parabolic SAR (AF 0.02→0.20). "
                "7 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _TREND_COLUMNS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = ["trend", "adx", "aroon", "parabolic_sar", "directional"],
        )
