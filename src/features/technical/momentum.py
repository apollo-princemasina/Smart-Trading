"""Momentum indicators: RSI, Stochastic, MACD, CCI, Williams %R, ROC, Momentum, TSI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_RSI_PERIOD   = 14
_STOCH_K      = 14
_STOCH_D      = 3
_MACD_FAST    = 12
_MACD_SLOW    = 26
_MACD_SIGNAL  = 9
_CCI_PERIOD   = 20
_WR_PERIOD    = 14
_ROC_PERIOD   = 12
_MOM_PERIOD   = 10
_TSI_LONG     = 25
_TSI_SHORT    = 13

_MOMENTUM_COLUMNS: list[str] = [
    "rsi", "stochastic_k", "stochastic_d",
    "macd", "macd_signal", "macd_histogram",
    "cci", "williams_r", "roc", "price_momentum", "tsi",
]


def _wilder_rma(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (RMA) via ewm with com=period-1."""
    return pd.Series(arr).ewm(com=period - 1, adjust=False).mean().to_numpy()


@FeatureRegistry.register
class MomentumEngine(BaseFeature):
    """RSI, Stochastic %K/%D, MACD/Signal/Histogram, CCI, Williams %R, ROC, Momentum, TSI."""

    name:             str       = "momentum"
    category:         str       = "technical"
    dependencies:     list[str] = []
    required_columns: list[str] = ["high", "low", "close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        high  = df["high"].to_numpy(dtype=np.float64)
        low   = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        close_s = pd.Series(close, index=df.index)

        # ── RSI (Wilder's, period 14) ──────────────────────────────────────────
        delta = np.diff(close, prepend=close[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_gain = _wilder_rma(gain, _RSI_PERIOD)
        avg_loss = _wilder_rma(loss, _RSI_PERIOD)
        rs   = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi  = np.where(avg_loss > 0, 100.0 - 100.0 / (1.0 + rs), 100.0)

        # ── Stochastic %K / %D ────────────────────────────────────────────────
        high_s = pd.Series(high, index=df.index)
        low_s  = pd.Series(low,  index=df.index)
        hh     = high_s.rolling(_STOCH_K, min_periods=1).max().to_numpy()
        ll     = low_s.rolling(_STOCH_K,  min_periods=1).min().to_numpy()
        denom  = np.where(hh - ll > 0, hh - ll, 1.0)
        stoch_k = np.where(hh - ll > 0, (close - ll) / denom * 100.0, 50.0)
        stoch_d = pd.Series(stoch_k, index=df.index).rolling(_STOCH_D, min_periods=1).mean().to_numpy()

        # ── MACD / Signal / Histogram ─────────────────────────────────────────
        ema_fast   = close_s.ewm(span=_MACD_FAST,   adjust=False).mean().to_numpy()
        ema_slow   = close_s.ewm(span=_MACD_SLOW,   adjust=False).mean().to_numpy()
        macd       = ema_fast - ema_slow
        macd_sig   = pd.Series(macd, index=df.index).ewm(span=_MACD_SIGNAL, adjust=False).mean().to_numpy()
        macd_hist  = macd - macd_sig

        # ── CCI ───────────────────────────────────────────────────────────────
        tp         = (high + low + close) / 3.0
        tp_s       = pd.Series(tp, index=df.index)
        tp_ma      = tp_s.rolling(_CCI_PERIOD, min_periods=1).mean()
        tp_md      = tp_s.rolling(_CCI_PERIOD, min_periods=1).apply(
                         lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci_denom  = (0.015 * tp_md).to_numpy()
        safe_denom = np.where(cci_denom > 0, cci_denom, 1.0)
        cci        = np.where(cci_denom > 0, (tp - tp_ma.to_numpy()) / safe_denom, 0.0)

        # ── Williams %R ───────────────────────────────────────────────────────
        hh_wr  = high_s.rolling(_WR_PERIOD, min_periods=1).max().to_numpy()
        ll_wr  = low_s.rolling(_WR_PERIOD,  min_periods=1).min().to_numpy()
        wr_den = np.where(hh_wr - ll_wr > 0, hh_wr - ll_wr, 1.0)
        williams_r = np.where(hh_wr - ll_wr > 0,
                              (hh_wr - close) / wr_den * -100.0, -50.0)

        # ── Rate of Change ────────────────────────────────────────────────────
        prev_roc  = close_s.shift(_ROC_PERIOD).to_numpy()
        safe_prev = np.where(prev_roc > 0, prev_roc, 1.0)
        roc       = np.where(prev_roc > 0, (close - prev_roc) / safe_prev * 100.0, 0.0)

        # ── Price Momentum (difference) — NaN-safe for warm-up bars ─────────────
        prev_mom       = close_s.shift(_MOM_PERIOD).to_numpy()
        price_momentum = np.where(np.isnan(prev_mom), 0.0, close - prev_mom)

        # ── TSI: True Strength Index ──────────────────────────────────────────
        m      = np.diff(close, prepend=close[0])
        abs_m  = np.abs(m)
        m_dbl  = pd.Series(m).ewm(span=_TSI_LONG, adjust=False).mean()
        m_dbl  = m_dbl.ewm(span=_TSI_SHORT, adjust=False).mean().to_numpy()
        am_dbl = pd.Series(abs_m).ewm(span=_TSI_LONG, adjust=False).mean()
        am_dbl = am_dbl.ewm(span=_TSI_SHORT, adjust=False).mean().to_numpy()
        tsi_den  = np.where(am_dbl > 0, am_dbl, 1.0)
        tsi      = np.where(am_dbl > 0, 100.0 * m_dbl / tsi_den, 0.0)

        out = pd.DataFrame(index=df.index)
        out["rsi"]             = rsi
        out["stochastic_k"]    = stoch_k
        out["stochastic_d"]    = stoch_d
        out["macd"]            = macd
        out["macd_signal"]     = macd_sig
        out["macd_histogram"]  = macd_hist
        out["cci"]             = cci
        out["williams_r"]      = williams_r
        out["roc"]             = roc
        out["price_momentum"]  = price_momentum
        out["tsi"]             = tsi
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "RSI-14, Stochastic %K/%D, MACD/Signal/Histogram, CCI-20, "
                "Williams %R-14, ROC-12, Price Momentum-10, TSI(25,13). "
                "11 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _MOMENTUM_COLUMNS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = ["momentum", "rsi", "macd", "stochastic", "cci", "tsi"],
        )
