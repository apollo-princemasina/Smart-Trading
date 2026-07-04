"""Composite technical engine — cross-indicator derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_TECH_COLUMNS: list[str] = [
    "price_vs_ema200",
    "price_vs_vwap",
    "macd_normalized",
    "rsi_stoch_divergence",
    "trend_strength",
]


@FeatureRegistry.register
class TechnicalEngine(BaseFeature):
    """Cross-indicator features derived from all five sub-engines."""

    name:             str       = "technical"
    category:         str       = "technical"
    dependencies:     list[str] = [
        "moving_averages", "momentum", "trend", "volatility", "oscillators"
    ]
    required_columns: list[str] = [
        "close",
        "ema200", "vwap",
        "macd", "atr",
        "rsi", "stochastic_k",
        "adx", "plus_di", "minus_di",
    ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["close"].to_numpy(dtype=np.float64)
        ema200 = df["ema200"].to_numpy(dtype=np.float64)
        vwap   = df["vwap"].to_numpy(dtype=np.float64)
        macd   = df["macd"].to_numpy(dtype=np.float64)
        atr    = df["atr"].to_numpy(dtype=np.float64)
        rsi    = df["rsi"].to_numpy(dtype=np.float64)
        stoch_k= df["stochastic_k"].to_numpy(dtype=np.float64)
        adx    = df["adx"].to_numpy(dtype=np.float64)
        plus_di= df["plus_di"].to_numpy(dtype=np.float64)
        minus_di=df["minus_di"].to_numpy(dtype=np.float64)

        # ── price_vs_ema200: % distance from EMA 200 ──────────────────────────
        safe_ema200 = np.where(ema200 > 0, ema200, 1.0)
        price_vs_ema200 = np.where(ema200 > 0,
                                   (close - ema200) / safe_ema200 * 100.0, 0.0)

        # ── price_vs_vwap: % distance from daily VWAP ─────────────────────────
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        price_vs_vwap = np.where(vwap > 0,
                                 (close - vwap) / safe_vwap * 100.0, 0.0)

        # ── macd_normalized: MACD relative to ATR ─────────────────────────────
        safe_atr = np.where(atr > 0, atr, 1.0)
        macd_normalized = np.where(atr > 0, macd / safe_atr, 0.0)

        # ── rsi_stoch_divergence: signed RSI-Stoch difference (normalised) ────
        rsi_norm   = (rsi   - 50.0) / 50.0
        stoch_norm = (stoch_k - 50.0) / 50.0
        rsi_stoch_divergence = rsi_norm - stoch_norm

        # ── trend_strength: ADX × DI direction sign ───────────────────────────
        di_sign      = np.sign(plus_di - minus_di)
        trend_strength = adx * di_sign / 100.0   # range [-1, 1]

        out = pd.DataFrame(index=df.index)
        out["price_vs_ema200"]      = price_vs_ema200
        out["price_vs_vwap"]        = price_vs_vwap
        out["macd_normalized"]      = macd_normalized
        out["rsi_stoch_divergence"] = rsi_stoch_divergence
        out["trend_strength"]       = trend_strength
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Cross-indicator composite features: price_vs_ema200, price_vs_vwap, "
                "macd_normalized (by ATR), rsi_stoch_divergence, trend_strength "
                "(ADX × DI direction sign). 5 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _TECH_COLUMNS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "low",
            tags       = ["composite", "cross_indicator", "technical"],
        )
