"""Premium and Discount Delta Volume — BigBeluga Pine Script translation.

Pairs every bar's volume with the ICT Premium/Discount framework to produce
institutional-grade buying/selling pressure features across two rolling windows:

  SR period    (50 bars)  — local / support-resistance view
  Macro period (200 bars) — broader institutional view

Delta Volume Formula (Pine Script faithful translation):
  neg_avg = rolling_mean(−volume on bearish bars, else 0, over N bars)
  pos_avg = rolling_mean(+volume on bullish bars, else 0, over N bars)
  delta   = clip( (neg_avg / pos_avg + 1) × 100, −100, 100 )

  delta > 0  →  net buying pressure in window
  delta = 0  →  balanced
  delta < 0  →  net selling pressure in window

Requires pd_zone from PremiumDiscountEngine to compute zone-attributed volumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_SR_PERIOD      = 50    # Pine Script "srPeriod" input
_MACRO_PERIOD   = 200   # Pine Script "macroPeriod" input
_TREND_FAST     = 5     # EMA span for fast delta — used by volume_trend
_TREND_SLOW     = 20    # EMA span for slow delta — used by volume_trend
_ACCEL_LOOKBACK = 5     # bars back for volume_acceleration


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).mean().to_numpy()


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).sum().to_numpy()


def _safe_delta(neg_avg: np.ndarray, pos_avg: np.ndarray) -> np.ndarray:
    """Compute the BigBeluga delta volume formula with safe division.

    Returns 0.0 when both averages are zero (no directional volume in window).
    """
    both_zero  = (pos_avg == 0.0) & (neg_avg == 0.0)
    safe_pos   = np.where(pos_avg != 0.0, pos_avg, 1e-8)
    raw        = np.clip((neg_avg / safe_pos + 1.0) * 100.0, -100.0, 100.0)
    return np.where(both_zero, 0.0, raw)


_OUTPUT_COLUMNS: list[str] = [
    "delta_volume",
    "delta_percent",
    "positive_volume",
    "negative_volume",
    "premium_volume",
    "discount_volume",
    "equilibrium_volume",
    "volume_imbalance",
    "buy_pressure",
    "sell_pressure",
    "macro_delta",
    "local_delta",
    "volume_strength",
    "volume_trend",
    "volume_acceleration",
    "volume_exhaustion",
    "volume_expansion",
    "volume_compression",
    "premium_strength",
    "discount_strength",
]


@FeatureRegistry.register
class PremiumDiscountDeltaVolumeEngine(BaseFeature):
    """BigBeluga-style delta volume split by ICT Premium / Discount zones.

    Two delta volumes are computed:
      - local_delta  / delta_volume : SR period (50 bars, Pine "srPeriod")
      - macro_delta                 : Macro period (200 bars, Pine "macroPeriod")

    Zone-attributed volumes (premium, discount, equilibrium) are derived
    from pd_zone produced by PremiumDiscountEngine — never recalculated here.

    All 20 output columns are float64 and ML-ready (no categoricals, no NaN).
    """

    name:             str       = "premium_discount_delta_volume"
    category:         str       = "volume"
    dependencies:     list[str] = ["premium_discount"]
    required_columns: list[str] = [
        "open", "high", "low", "close", "volume",
        "pd_zone",
    ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0914
        open_arr  = df["open"].to_numpy(dtype=np.float64)
        high_arr  = df["high"].to_numpy(dtype=np.float64)
        low_arr   = df["low"].to_numpy(dtype=np.float64)
        close_arr = df["close"].to_numpy(dtype=np.float64)
        vol_arr   = df["volume"].to_numpy(dtype=np.float64)
        pd_zone   = np.nan_to_num(df["pd_zone"].to_numpy(dtype=np.float64), nan=0.0)

        # ── Directional volume decomposition ──────────────────────────────────
        bull_vol = np.where(close_arr > open_arr, vol_arr, 0.0)  # bullish bars
        bear_vol = np.where(close_arr < open_arr, vol_arr, 0.0)  # bearish bars (abs)
        neg_vol  = -bear_vol                                      # sign for delta formula

        # ── SR period (50) rolling averages ───────────────────────────────────
        pos_avg_sr  = _rolling_mean(bull_vol, _SR_PERIOD)
        neg_avg_sr  = _rolling_mean(neg_vol,  _SR_PERIOD)
        bull_sum_sr = _rolling_sum(bull_vol,  _SR_PERIOD)
        bear_sum_sr = _rolling_sum(bear_vol,  _SR_PERIOD)

        # ── Macro period (200) rolling averages ───────────────────────────────
        pos_avg_mac  = _rolling_mean(bull_vol, _MACRO_PERIOD)
        neg_avg_mac  = _rolling_mean(neg_vol,  _MACRO_PERIOD)
        vol_ma_mac   = _rolling_mean(vol_arr,  _MACRO_PERIOD)
        vol_ma_sr    = _rolling_mean(vol_arr,  _SR_PERIOD)

        # ── 1. delta_volume  (Pine faithful SR delta, −100 … +100) ───────────
        delta_volume = _safe_delta(neg_avg_sr, pos_avg_sr)

        # ── 2. delta_percent  (normalised to −1 … +1) ────────────────────────
        delta_percent = delta_volume / 100.0

        # ── 3. positive_volume  (mean bull vol per bar in SR window) ─────────
        positive_volume = pos_avg_sr

        # ── 4. negative_volume  (mean bear vol per bar in SR window, abs) ────
        negative_volume = _rolling_mean(bear_vol, _SR_PERIOD)

        # ── 5-7. Zone-attributed mean volumes ────────────────────────────────
        premium_volume     = _rolling_mean(
            np.where(pd_zone == 1.0,  vol_arr, 0.0), _SR_PERIOD)
        discount_volume    = _rolling_mean(
            np.where(pd_zone == -1.0, vol_arr, 0.0), _SR_PERIOD)
        equilibrium_volume = _rolling_mean(
            np.where(pd_zone == 0.0,  vol_arr, 0.0), _SR_PERIOD)

        # ── 8-10. Pressure / imbalance ────────────────────────────────────────
        dir_total = bull_sum_sr + bear_sum_sr
        safe_dir  = np.where(dir_total > 0, dir_total, 1e-8)

        volume_imbalance = np.where(
            dir_total > 0,
            (bull_sum_sr - bear_sum_sr) / safe_dir,
            0.0,
        )  # [-1, 1]
        buy_pressure  = np.where(dir_total > 0, bull_sum_sr / safe_dir, 0.5)   # [0, 1]
        sell_pressure = np.where(dir_total > 0, bear_sum_sr / safe_dir, 0.5)   # [0, 1]

        # ── 11. macro_delta  (Pine faithful macro delta, −100 … +100) ────────
        macro_delta = _safe_delta(neg_avg_mac, pos_avg_mac)

        # ── 12. local_delta  (SR alias — matches Pine's deltaVolSR naming) ───
        local_delta = delta_volume.copy()

        # ── 13. volume_strength  (relative to macro baseline) ────────────────
        volume_strength = vol_arr / (vol_ma_mac + 1e-8)

        # ── 14. volume_trend  (EMA crossover of delta, −1 / 0 / +1) ─────────
        delta_s   = pd.Series(delta_volume)
        ema_fast  = delta_s.ewm(span=_TREND_FAST, adjust=False).mean().to_numpy()
        ema_slow  = delta_s.ewm(span=_TREND_SLOW, adjust=False).mean().to_numpy()
        volume_trend = np.sign(ema_fast - ema_slow)

        # ── 15. volume_acceleration  (delta rate-of-change over 5 bars) ──────
        d_series = pd.Series(delta_volume)
        volume_acceleration = (
            d_series - d_series.shift(_ACCEL_LOOKBACK)
        ).fillna(0.0).to_numpy()

        # ── 16. volume_exhaustion  (high effort, low price efficiency) ────────
        body          = np.abs(close_arr - open_arr)
        candle_range  = np.where(high_arr > low_arr, high_arr - low_arr, 1e-8)
        price_eff     = body / candle_range       # 1.0 = full-body candle
        vol_rel       = vol_arr / (vol_ma_mac + 1e-8)
        volume_exhaustion = np.clip(vol_rel * (1.0 - price_eff), 0.0, None)

        # ── 17-18. Expansion / compression  (vs SR baseline) ─────────────────
        rel_sr = vol_arr / (vol_ma_sr + 1e-8)
        volume_expansion   = np.clip(rel_sr - 1.0, 0.0, None)
        volume_compression = np.clip(1.0 - rel_sr, 0.0, None)

        # ── 19. premium_strength  (% bearish volume inside premium zone) ─────
        pb_bull = _rolling_sum(np.where(pd_zone == 1.0, bull_vol, 0.0), _SR_PERIOD)
        pb_bear = _rolling_sum(np.where(pd_zone == 1.0, bear_vol, 0.0), _SR_PERIOD)
        pb_tot  = pb_bull + pb_bear
        safe_pb = np.where(pb_tot > 0, pb_tot, 1.0)
        premium_strength = np.where(
            pb_tot > 0, pb_bear / safe_pb * 100.0, 50.0)

        # ── 20. discount_strength  (% bullish volume inside discount zone) ───
        db_bull = _rolling_sum(np.where(pd_zone == -1.0, bull_vol, 0.0), _SR_PERIOD)
        db_bear = _rolling_sum(np.where(pd_zone == -1.0, bear_vol, 0.0), _SR_PERIOD)
        db_tot  = db_bull + db_bear
        safe_db = np.where(db_tot > 0, db_tot, 1.0)
        discount_strength = np.where(
            db_tot > 0, db_bull / safe_db * 100.0, 50.0)

        # ── Assemble output ───────────────────────────────────────────────────
        out = pd.DataFrame(index=df.index)
        out["delta_volume"]        = delta_volume
        out["delta_percent"]       = delta_percent
        out["positive_volume"]     = positive_volume
        out["negative_volume"]     = negative_volume
        out["premium_volume"]      = premium_volume
        out["discount_volume"]     = discount_volume
        out["equilibrium_volume"]  = equilibrium_volume
        out["volume_imbalance"]    = volume_imbalance
        out["buy_pressure"]        = buy_pressure
        out["sell_pressure"]       = sell_pressure
        out["macro_delta"]         = macro_delta
        out["local_delta"]         = local_delta
        out["volume_strength"]     = volume_strength
        out["volume_trend"]        = volume_trend
        out["volume_acceleration"] = volume_acceleration
        out["volume_exhaustion"]   = volume_exhaustion
        out["volume_expansion"]    = volume_expansion
        out["volume_compression"]  = volume_compression
        out["premium_strength"]    = premium_strength
        out["discount_strength"]   = discount_strength

        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "BigBeluga Premium & Discount Delta Volume.  Translates two "
                "rolling-window delta volume calculations (SR=50, Macro=200 bars) "
                "into 20 ML-ready features covering buying/selling pressure, "
                "zone-attributed volumes, and volume regime signals."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _OUTPUT_COLUMNS,
            version          = "1.0.0",
            author           = "Smart Trading Team",
            complexity       = "low",
            tags             = [
                "ICT", "smart_money", "delta_volume", "BigBeluga",
                "premium", "discount", "volume_profile",
                "buying_pressure", "selling_pressure",
            ],
        )
