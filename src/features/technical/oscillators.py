"""Volume/price oscillators: VWAP, VWMA, OBV, CMF, MFI, AD, Force Index, EOM."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_CMF_PERIOD = 20
_MFI_PERIOD = 14
_EOM_PERIOD = 14

_OSCILLATOR_COLUMNS: list[str] = [
    "vwap", "vwma",
    "obv", "cmf", "mfi", "ad",
    "force_index", "eom",
]


@FeatureRegistry.register
class OscillatorsEngine(BaseFeature):
    """VWAP (daily reset), VWMA, OBV, CMF, MFI, Accumulation/Distribution, Force Index, EOM."""

    name:             str       = "oscillators"
    category:         str       = "technical"
    dependencies:     list[str] = []
    required_columns: list[str] = ["high", "low", "close", "volume"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        high   = df["high"].to_numpy(dtype=np.float64)
        low    = df["low"].to_numpy(dtype=np.float64)
        close  = df["close"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

        tp = (high + low + close) / 3.0

        # ── Daily-reset VWAP ──────────────────────────────────────────────────
        if hasattr(df.index, "date"):
            date_key = df.index.date
        else:
            date_key = pd.to_datetime(df.index).date
        dates = pd.array(date_key, dtype="object")
        tp_s   = pd.Series(tp * volume, index=df.index)
        vol_s  = pd.Series(volume,      index=df.index)
        date_s = pd.Series(dates,       index=df.index)
        cum_tp  = tp_s.groupby(date_s).cumsum().to_numpy()
        cum_vol = vol_s.groupby(date_s).cumsum().to_numpy()
        safe_cv = np.where(cum_vol > 0, cum_vol, 1.0)
        vwap    = np.where(cum_vol > 0, cum_tp / safe_cv, tp)

        # ── VWMA (volume-weighted MA, 20-bar) ─────────────────────────────────
        tv = tp * volume
        tv_s = pd.Series(tv, index=df.index)
        vwma = (tv_s.rolling(20, min_periods=1).sum() /
                pd.Series(volume, index=df.index).rolling(20, min_periods=1).sum().replace(0, np.nan)
                ).fillna(pd.Series(tp, index=df.index)).to_numpy()

        # ── OBV ───────────────────────────────────────────────────────────────
        prev_close = np.concatenate([[close[0]], close[:-1]])
        direction  = np.sign(close - prev_close)
        obv        = np.cumsum(direction * volume)

        # ── Accumulation / Distribution ────────────────────────────────────────
        hl_range  = high - low
        safe_hl   = np.where(hl_range > 0, hl_range, 1.0)
        clv       = np.where(hl_range > 0, ((close - low) - (high - close)) / safe_hl, 0.0)
        ad        = np.cumsum(clv * volume)

        # ── Chaikin Money Flow ────────────────────────────────────────────────
        clv_vol  = clv * volume
        cv_s     = pd.Series(clv_vol, index=df.index)
        vol_s2   = pd.Series(volume,  index=df.index)
        cmf_num  = cv_s.rolling(_CMF_PERIOD,  min_periods=1).sum().to_numpy()
        cmf_den  = vol_s2.rolling(_CMF_PERIOD, min_periods=1).sum().to_numpy()
        safe_den = np.where(cmf_den > 0, cmf_den, 1.0)
        cmf      = np.where(cmf_den > 0, cmf_num / safe_den, 0.0)

        # ── Money Flow Index ───────────────────────────────────────────────────
        prev_tp  = np.concatenate([[tp[0]], tp[:-1]])
        pos_mf   = np.where(tp > prev_tp, tp * volume, 0.0)
        neg_mf   = np.where(tp < prev_tp, tp * volume, 0.0)
        pos_mf_s = pd.Series(pos_mf, index=df.index)
        neg_mf_s = pd.Series(neg_mf, index=df.index)
        pos_sum  = pos_mf_s.rolling(_MFI_PERIOD, min_periods=1).sum().to_numpy()
        neg_sum  = neg_mf_s.rolling(_MFI_PERIOD, min_periods=1).sum().to_numpy()
        mfr      = np.where(neg_sum > 0, pos_sum / neg_sum, 100.0)
        mfi      = np.where(neg_sum > 0, 100.0 - 100.0 / (1.0 + mfr), 100.0)

        # ── Force Index ───────────────────────────────────────────────────────
        force_index = (close - prev_close) * volume

        # ── Ease of Movement ──────────────────────────────────────────────────
        prev_high = np.concatenate([[high[0]], high[:-1]])
        prev_low  = np.concatenate([[low[0]],  low[:-1]])
        midpoint_move = ((high + low) / 2.0) - ((prev_high + prev_low) / 2.0)
        box_ratio     = np.where(high - low > 0,
                                 (volume / 1e6) / (high - low), 0.0)
        safe_box = np.where(box_ratio > 0, box_ratio, 1.0)
        raw_eom  = np.where((high - low > 0) & (box_ratio > 0),
                            midpoint_move / safe_box, 0.0)
        eom      = pd.Series(raw_eom, index=df.index).rolling(
                       _EOM_PERIOD, min_periods=1).mean().to_numpy()

        out = pd.DataFrame(index=df.index)
        out["vwap"]        = vwap
        out["vwma"]        = vwma
        out["obv"]         = obv
        out["cmf"]         = cmf
        out["mfi"]         = mfi
        out["ad"]          = ad
        out["force_index"] = force_index
        out["eom"]         = eom
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "VWAP (daily reset), VWMA-20, OBV, CMF-20, MFI-14, "
                "Accumulation/Distribution, Force Index, EOM-14. "
                "8 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _OSCILLATOR_COLUMNS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = ["oscillator", "volume", "vwap", "obv", "cmf", "mfi", "ad"],
        )
