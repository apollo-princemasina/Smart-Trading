"""Volatility indicators: ATR, Bollinger Bands, Keltner Channels, Donchian, Chaikin Volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_ATR_PERIOD    = 14
_BB_PERIOD     = 20
_BB_STD        = 2.0
_KC_PERIOD     = 20
_KC_MULT       = 1.5
_DC_PERIOD     = 20
_CHAIKIN_EMA   = 10   # EMA applied to H-L range for Chaikin Volatility
_CHAIKIN_ROC   = 10   # ROC period for Chaikin Volatility

_VOLATILITY_COLUMNS: list[str] = [
    "atr", "normalized_atr",
    "bb_upper", "bb_lower", "bb_width", "bb_percent_b",
    "kc_upper", "kc_lower",
    "dc_upper", "dc_lower",
    "chaikin_volatility",
]


def _wilder_rma(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).ewm(com=period - 1, adjust=False).mean().to_numpy()


@FeatureRegistry.register
class VolatilityEngine(BaseFeature):
    """ATR/NormATR, Bollinger Bands, Keltner Channels, Donchian Channels, Chaikin Volatility."""

    name:             str       = "volatility"
    category:         str       = "technical"
    dependencies:     list[str] = []
    required_columns: list[str] = ["high", "low", "close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        high  = df["high"].to_numpy(dtype=np.float64)
        low   = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        close_s = pd.Series(close, index=df.index)
        high_s  = pd.Series(high,  index=df.index)
        low_s   = pd.Series(low,   index=df.index)

        prev_close = np.concatenate([[close[0]], close[:-1]])

        # ── True Range & ATR ──────────────────────────────────────────────────
        tr  = np.maximum(high - low,
              np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = _wilder_rma(tr, _ATR_PERIOD)

        norm_atr = np.where(close > 0, atr / close * 100.0, 0.0)

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_mid   = close_s.rolling(_BB_PERIOD, min_periods=1).mean()
        bb_std   = close_s.rolling(_BB_PERIOD, min_periods=1).std(ddof=0)
        bb_upper = (bb_mid + _BB_STD * bb_std).to_numpy()
        bb_lower = (bb_mid - _BB_STD * bb_std).to_numpy()
        bb_width_arr = bb_upper - bb_lower

        bb_mid_np = bb_mid.to_numpy()
        safe_bw  = np.where(bb_mid_np > 0, bb_mid_np, 1.0)
        bb_width = np.where(bb_mid_np > 0, bb_width_arr / safe_bw * 100.0, 0.0)

        safe_bb_range = np.where(bb_width_arr > 0, bb_width_arr, 1.0)
        bb_percent_b  = np.where(bb_width_arr > 0,
                                 (close - bb_lower) / safe_bb_range, 0.5)

        # ── Keltner Channels ──────────────────────────────────────────────────
        ema_kc   = close_s.ewm(span=_KC_PERIOD, adjust=False).mean().to_numpy()
        kc_upper = ema_kc + _KC_MULT * atr
        kc_lower = ema_kc - _KC_MULT * atr

        # ── Donchian Channels ─────────────────────────────────────────────────
        dc_upper = high_s.rolling(_DC_PERIOD, min_periods=1).max().to_numpy()
        dc_lower = low_s.rolling(_DC_PERIOD,  min_periods=1).min().to_numpy()

        # ── Chaikin Volatility: EMA(H-L) ROC ─────────────────────────────────
        hl       = high - low
        hl_ema   = pd.Series(hl, index=df.index).ewm(span=_CHAIKIN_EMA, adjust=False).mean()
        hl_prev  = hl_ema.shift(_CHAIKIN_ROC).to_numpy()
        hl_ema_np= hl_ema.to_numpy()
        safe_prev= np.where(hl_prev > 0, hl_prev, 1.0)
        chaikin_volatility = np.where(hl_prev > 0,
                                      (hl_ema_np - hl_prev) / safe_prev * 100.0, 0.0)

        out = pd.DataFrame(index=df.index)
        out["atr"]               = atr
        out["normalized_atr"]    = norm_atr
        out["bb_upper"]          = bb_upper
        out["bb_lower"]          = bb_lower
        out["bb_width"]          = bb_width
        out["bb_percent_b"]      = bb_percent_b
        out["kc_upper"]          = kc_upper
        out["kc_lower"]          = kc_lower
        out["dc_upper"]          = dc_upper
        out["dc_lower"]          = dc_lower
        out["chaikin_volatility"]= chaikin_volatility
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "ATR-14 and Normalized ATR; Bollinger Bands-20 (width, %B); "
                "Keltner Channels (EMA20, 1.5×ATR); Donchian Channels-20; "
                "Chaikin Volatility (EMA10 of H-L, ROC10). 11 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _VOLATILITY_COLUMNS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = ["volatility", "atr", "bollinger", "keltner", "donchian"],
        )
