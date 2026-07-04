"""Cross-timeframe OHLCV consistency validator.

Aggregates the lower timeframe to the higher timeframe's resolution
and compares open / high / low / close / volume. Any mismatch beyond
floating-point tolerance is flagged.

Supported pairs checked by the pipeline:
    M15  ->  H1   (4 M15 candles per H1)
    H1   ->  H4   (4 H1  candles per H4)
    H4   ->  D1   (6 H4  candles per D1, forex-day = 24 h)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Aggregation target frequency for each higher timeframe.
_RESAMPLE_FREQ: dict[str, str] = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}

# Tolerance for floating-point price comparison (1 micro-pip for EURUSD 5-dp).
_PRICE_TOL = 1e-5


@dataclass
class CrossTFResult:
    """Result for one (lower, higher) timeframe pair."""

    lower_tf:  str
    higher_tf: str
    periods_compared: int = 0
    open_mismatches:  int = 0
    high_mismatches:  int = 0
    low_mismatches:   int = 0
    close_mismatches: int = 0
    volume_mismatches: int = 0
    incomplete_periods: int = 0  # periods where lower TF has missing candles
    issues:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return len(self.issues) == 0


class TimeframeValidator:
    """
    Validate OHLCV consistency between a lower and higher timeframe.

    Aggregates the lower TF and compares against the higher TF by
    timestamp. Only periods present in BOTH datasets are compared —
    missing candles are counted as incomplete periods but are not a
    hard failure.
    """

    def validate(
        self,
        lower_df:  pd.DataFrame,
        higher_df: pd.DataFrame,
        lower_tf:  str,
        higher_tf: str,
    ) -> CrossTFResult:
        result = CrossTFResult(lower_tf=lower_tf, higher_tf=higher_tf)

        if higher_tf not in _RESAMPLE_FREQ:
            result.warnings.append(
                f"No resample rule defined for {higher_tf}; skipping cross-TF check."
            )
            return result

        if lower_df.empty or higher_df.empty:
            result.warnings.append("One or both DataFrames are empty; skipping.")
            return result

        agg = self._aggregate(lower_df, _RESAMPLE_FREQ[higher_tf])
        if agg.empty:
            result.warnings.append("Aggregation produced no data; skipping.")
            return result

        self._compare(agg, higher_df, result)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample a lower-TF DataFrame up to `freq`."""
        ts = df["timestamp"]
        if getattr(ts.dtype, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC")

        tmp = df.copy()
        tmp = tmp.set_index(ts)

        agg = tmp.resample(freq, label="left", closed="left").agg(
            open  =("open",        "first"),
            high  =("high",        "max"),
            low   =("low",         "min"),
            close =("close",       "last"),
            tick_volume=("tick_volume", "sum"),
            candle_count=("open",  "count"),
        )
        agg = agg.dropna(subset=["open", "close"])
        agg.index.name = "timestamp"
        return agg.reset_index()

    @staticmethod
    def _compare(
        agg:       pd.DataFrame,
        higher_df: pd.DataFrame,
        result:    CrossTFResult,
    ) -> None:
        ts_col = "timestamp"

        # Align on UTC timestamps
        h = higher_df.copy()
        h_ts = h[ts_col]
        if getattr(h_ts.dtype, "tz", None) is None:
            h[ts_col] = h_ts.dt.tz_localize("UTC")

        a_ts = agg[ts_col]
        if getattr(a_ts.dtype, "tz", None) is None:
            agg[ts_col] = a_ts.dt.tz_localize("UTC")

        merged = pd.merge(
            agg[["timestamp", "open", "high", "low", "close",
                 "tick_volume", "candle_count"]],
            h[["timestamp", "open", "high", "low", "close", "tick_volume"]],
            on="timestamp",
            how="inner",
            suffixes=("_agg", "_htf"),
        )

        result.periods_compared = len(merged)
        if result.periods_compared == 0:
            result.warnings.append(
                "No overlapping timestamps found between aggregated lower TF "
                "and the higher TF. Check timeframe alignment."
            )
            return

        # Count periods where the lower TF had missing candles
        expected_count = {"H1": 4, "H4": 4, "D1": 6, "W1": 5}.get(
            result.higher_tf, None
        )
        if expected_count:
            incomplete = int(
                (merged["candle_count"] < expected_count).sum()
            )
            result.incomplete_periods = incomplete
            if incomplete:
                result.warnings.append(
                    f"{incomplete} periods have fewer lower-TF candles than "
                    f"expected ({expected_count}). "
                    "Missing candles cause slight OHLC deviation."
                )

        # Price comparisons
        for col in ("open", "high", "low", "close"):
            diff = (merged[f"{col}_agg"] - merged[f"{col}_htf"]).abs()
            bad  = int((diff > _PRICE_TOL).sum())
            setattr(result, f"{col}_mismatches", bad)
            if bad:
                sample = merged.loc[diff > _PRICE_TOL, "timestamp"].iloc[:3].tolist()
                result.issues.append(
                    f"{bad} {col} mismatches between "
                    f"aggregated {result.lower_tf} and {result.higher_tf}. "
                    f"Sample timestamps: {sample}"
                )

        # Volume comparison (lenient — tick_volume doesn't aggregate perfectly)
        v_diff = (merged["tick_volume_agg"] - merged["tick_volume_htf"]).abs()
        v_pct  = v_diff / merged["tick_volume_htf"].clip(lower=1)
        v_bad  = int((v_pct > 0.10).sum())  # >10% relative difference
        result.volume_mismatches = v_bad
        if v_bad:
            result.warnings.append(
                f"{v_bad} periods with tick_volume deviation > 10% between "
                f"aggregated {result.lower_tf} and {result.higher_tf}. "
                "Tick_volume sampling differs per timeframe in MT5."
            )
