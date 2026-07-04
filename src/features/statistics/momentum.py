"""Momentum statistics: velocity, acceleration, persistence, trend continuity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_VEL_WINDOW   = 5
_MOM_SHORT    = 5
_MOM_LONG     = 20
_PERSIST_WIN  = 20   # autocorrelation window for momentum_persistence
_TREND_WIN    = 20   # fraction of bars with same-sign return

_MOMENTUM_STAT_COLS: list[str] = [
    "price_velocity",
    "price_acceleration",
    "price_deceleration",
    "rolling_momentum_5",
    "rolling_momentum_20",
    "momentum_persistence",
    "trend_persistence",
]


def _autocorr_lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of x (Pearson, returns 0.0 on constant series)."""
    if len(x) < 3:
        return 0.0
    x1, x2 = x[:-1], x[1:]
    mu1, mu2 = np.mean(x1), np.mean(x2)
    s1 = np.std(x1, ddof=1)
    s2 = np.std(x2, ddof=1)
    if s1 < 1e-10 or s2 < 1e-10:
        return 0.0
    return float(np.mean((x1 - mu1) * (x2 - mu2)) / (s1 * s2))


@FeatureRegistry.register
class MomentumStatisticsEngine(BaseFeature):
    """Price velocity, acceleration, deceleration, rolling momentum, persistence."""

    name:             str       = "momentum_stats"
    category:         str       = "statistics"
    dependencies:     list[str] = ["returns"]
    required_columns: list[str] = ["close", "log_return"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        close   = df["close"].to_numpy(dtype=np.float64)
        log_ret = df["log_return"].to_numpy(dtype=np.float64)
        lr_s    = pd.Series(log_ret, index=df.index)

        # ── Price velocity: mean log return over last 5 bars ──────────────────
        price_velocity = lr_s.rolling(_VEL_WINDOW, min_periods=1).mean().to_numpy()

        # ── Acceleration / deceleration ───────────────────────────────────────
        pv_s               = pd.Series(price_velocity, index=df.index)
        prev_velocity      = pv_s.shift(1).fillna(0.0).to_numpy()
        price_acceleration = price_velocity - prev_velocity
        price_deceleration = np.maximum(0.0, -price_acceleration)  # >0 only when slowing

        # ── Rolling momentum sums ─────────────────────────────────────────────
        rolling_momentum_5  = lr_s.rolling(_MOM_SHORT, min_periods=1).sum().to_numpy()
        rolling_momentum_20 = lr_s.rolling(_MOM_LONG,  min_periods=1).sum().to_numpy()

        # ── Momentum persistence: rolling lag-1 autocorrelation of log_return ─
        momentum_persistence = lr_s.rolling(_PERSIST_WIN, min_periods=4).apply(
            _autocorr_lag1, raw=True).to_numpy()
        momentum_persistence = np.where(
            np.isnan(momentum_persistence), 0.0, momentum_persistence)

        # ── Trend persistence: fraction of bars with same-sign return ─────────
        same_sign = (np.sign(log_ret[1:]) == np.sign(log_ret[:-1])).astype(np.float64)
        same_sign = np.concatenate([[0.0], same_sign])
        trend_persistence = pd.Series(same_sign, index=df.index).rolling(
            _TREND_WIN, min_periods=1).mean().to_numpy()

        out = pd.DataFrame(index=df.index)
        out["price_velocity"]     = price_velocity
        out["price_acceleration"] = price_acceleration
        out["price_deceleration"] = price_deceleration
        out["rolling_momentum_5"] = rolling_momentum_5
        out["rolling_momentum_20"]= rolling_momentum_20
        out["momentum_persistence"]= momentum_persistence
        out["trend_persistence"]  = trend_persistence
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Price velocity (5-bar mean log return), acceleration "
                "(1-bar change in velocity), deceleration (positive when slowing), "
                "rolling momentum sums (5/20), lag-1 autocorrelation of returns "
                "(momentum persistence), and same-sign fraction (trend persistence). "
                "7 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _MOMENTUM_STAT_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "medium",
            tags       = ["momentum", "velocity", "acceleration", "persistence",
                          "statistics"],
        )
