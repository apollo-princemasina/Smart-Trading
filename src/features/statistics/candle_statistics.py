"""Per-bar candle anatomy: body, wicks, range, pattern flags, sequence counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

_SEQ_WINDOW = 10   # rolling window for higher/lower counts

_CANDLE_COLS: list[str] = [
    "body_size", "body_ratio",
    "upper_wick", "lower_wick", "upper_wick_ratio", "lower_wick_ratio",
    "total_range", "true_range", "body_to_range_ratio",
    "is_bullish", "is_bearish",
    "doji_score", "marubozu_score",
    "inside_bar", "outside_bar",
    "consecutive_bulls", "consecutive_bears",
    "higher_close_count", "lower_close_count",
    "higher_high_count", "lower_low_count",
]


def _consecutive_count(condition: np.ndarray) -> np.ndarray:
    """Vectorised count of consecutive True values ending at each position."""
    s      = pd.Series(condition.astype(np.int8))
    cumsum = s.cumsum()
    reset  = cumsum.where(~s.astype(bool)).ffill().fillna(0)
    return (cumsum - reset).to_numpy(dtype=np.float64)


@FeatureRegistry.register
class CandleStatisticsEngine(BaseFeature):
    """Body/wick anatomy, doji/marubozu scores, inside/outside bars, sequence counts."""

    name:             str       = "candle_statistics"
    category:         str       = "statistics"
    dependencies:     list[str] = []
    required_columns: list[str] = ["open", "high", "low", "close"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        open_  = df["open"].to_numpy(dtype=np.float64)
        high   = df["high"].to_numpy(dtype=np.float64)
        low    = df["low"].to_numpy(dtype=np.float64)
        close  = df["close"].to_numpy(dtype=np.float64)

        prev_close = np.concatenate([[close[0]], close[:-1]])
        prev_high  = np.concatenate([[high[0]],  high[:-1]])
        prev_low   = np.concatenate([[low[0]],   low[:-1]])

        # ── Body and wick geometry ─────────────────────────────────────────────
        body_top    = np.maximum(open_, close)
        body_bot    = np.minimum(open_, close)
        body_size   = body_top - body_bot
        upper_wick  = high - body_top
        lower_wick  = body_bot - low
        total_range = high - low

        safe_range = np.where(total_range > 0, total_range, 1.0)
        body_ratio        = np.where(total_range > 0, body_size  / safe_range, 0.0)
        upper_wick_ratio  = np.where(total_range > 0, upper_wick / safe_range, 0.0)
        lower_wick_ratio  = np.where(total_range > 0, lower_wick / safe_range, 0.0)

        # ── True Range ────────────────────────────────────────────────────────
        true_range = np.maximum(total_range,
                     np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

        safe_tr = np.where(true_range > 0, true_range, 1.0)
        body_to_range_ratio = np.where(true_range > 0, body_size / safe_tr, 0.0)

        # ── Directional flags ─────────────────────────────────────────────────
        is_bullish = (close > open_).astype(np.float64)
        is_bearish = (close < open_).astype(np.float64)

        # ── Pattern scores ────────────────────────────────────────────────────
        doji_score     = 1.0 - body_ratio                          # 1=perfect doji
        marubozu_score = (body_ratio
                          * (1.0 - upper_wick_ratio)
                          * (1.0 - lower_wick_ratio))              # 1=no-wick candle

        # ── Inside / Outside bar ──────────────────────────────────────────────
        inside_bar  = ((high < prev_high) & (low > prev_low)).astype(np.float64)
        outside_bar = ((high > prev_high) & (low < prev_low)).astype(np.float64)

        # ── Consecutive runs ──────────────────────────────────────────────────
        consecutive_bulls = _consecutive_count(is_bullish.astype(bool))
        consecutive_bears = _consecutive_count(is_bearish.astype(bool))

        # ── Rolling directional counts (10-bar) ───────────────────────────────
        higher_close = (close > prev_close).astype(np.float64)
        lower_close  = (close < prev_close).astype(np.float64)
        higher_high  = (high > prev_high).astype(np.float64)
        lower_low    = (low  < prev_low).astype(np.float64)

        def _roll_sum(arr: np.ndarray) -> np.ndarray:
            return pd.Series(arr, index=df.index).rolling(
                _SEQ_WINDOW, min_periods=1).sum().to_numpy()

        out = pd.DataFrame(index=df.index)
        out["body_size"]          = body_size
        out["body_ratio"]         = body_ratio
        out["upper_wick"]         = upper_wick
        out["lower_wick"]         = lower_wick
        out["upper_wick_ratio"]   = upper_wick_ratio
        out["lower_wick_ratio"]   = lower_wick_ratio
        out["total_range"]        = total_range
        out["true_range"]         = true_range
        out["body_to_range_ratio"]= body_to_range_ratio
        out["is_bullish"]         = is_bullish
        out["is_bearish"]         = is_bearish
        out["doji_score"]         = doji_score
        out["marubozu_score"]     = marubozu_score
        out["inside_bar"]         = inside_bar
        out["outside_bar"]        = outside_bar
        out["consecutive_bulls"]  = consecutive_bulls
        out["consecutive_bears"]  = consecutive_bears
        out["higher_close_count"] = _roll_sum(higher_close)
        out["lower_close_count"]  = _roll_sum(lower_close)
        out["higher_high_count"]  = _roll_sum(higher_high)
        out["lower_low_count"]    = _roll_sum(lower_low)
        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "Body/wick anatomy (size, ratio, wick ratios), true range, "
                "doji/marubozu scores, inside/outside bar flags, consecutive "
                "bull/bear run lengths, and 10-bar rolling directional counts. "
                "21 float64 columns."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _CANDLE_COLS,
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "low",
            tags       = ["candle", "body", "wick", "pattern", "sequence", "statistics"],
        )
