"""Forex market calendar validation — weekend filtering and gap classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Forex markets close Friday ~22:00 UTC and reopen Sunday ~22:00 UTC.
# A gap across this window is expected. We allow 72 h max for the weekend
# gap (covers DST shifts and broker-specific open times).
_WEEKEND_MIN_GAP = pd.Timedelta(hours=44)   # Fri 22:00 → Sun 22:00 = 48 h
_WEEKEND_MAX_GAP = pd.Timedelta(hours=76)   # extra slack for public holidays

# Gap larger than this times the candle interval AND not a weekend gap is
# flagged as unexpected (e.g. broker outage, missing data).
_INTRA_WEEK_MULTIPLIER = 4

# Weekday integers: Monday=0 … Sunday=6
_SATURDAY = 5
_SUNDAY   = 6

# Known major Forex holidays (month, day) — markets are thin but often open.
# We record these so they don't inflate the "unexpected gap" count.
_THIN_MARKET_DATES: set[tuple[int, int]] = {
    (1, 1),   # New Year's Day
    (12, 25), # Christmas Day
    (12, 26), # Boxing Day
}

_FREQ_MAP: dict[str, pd.Timedelta] = {
    "M15": pd.Timedelta(minutes=15),
    "H1":  pd.Timedelta(hours=1),
    "H4":  pd.Timedelta(hours=4),
    "D1":  pd.Timedelta(days=1),
    "W1":  pd.Timedelta(weeks=1),
}


@dataclass
class CalendarReport:
    """Summary of market-calendar checks for one timeframe."""

    timeframe:         str
    total_rows:        int
    weekend_candles:   int = 0
    expected_gaps:     int = 0   # Weekend / holiday gaps
    unexpected_gaps:   int = 0   # Mid-week gaps > threshold
    thin_market_rows:  int = 0   # Rows on holidays (thin liquidity)
    gap_details:       list[dict] = field(default_factory=list)
    warnings:          list[str]  = field(default_factory=list)


class ForexCalendar:
    """
    Classify candles and gaps against the standard Forex trading week.

    Rules:
      - Saturday candles are never valid; Sunday candles before 21:00 UTC
        are suspicious (market opens ~22:00 UTC).
      - A gap from Friday to Monday (44–76 h) is expected.
      - Public-holiday gaps within a trading week (< 44 h) are noted but
        not flagged as errors.
      - Any other gap > 4× the expected interval is flagged unexpected.
    """

    def validate(self, df: pd.DataFrame, timeframe: str) -> CalendarReport:
        report = CalendarReport(timeframe=timeframe, total_rows=len(df))

        if df.empty or "timestamp" not in df.columns:
            report.warnings.append("Empty DataFrame — calendar validation skipped.")
            return report

        ts = df["timestamp"].dt.tz_convert("UTC") if df["timestamp"].dt.tz else df["timestamp"]

        self._check_weekend_candles(ts, report)
        self._check_thin_market_candles(ts, report)

        if timeframe in _FREQ_MAP:
            self._classify_gaps(ts, timeframe, report)

        return report

    # ------------------------------------------------------------------

    @staticmethod
    def _check_weekend_candles(ts: pd.Series, report: CalendarReport) -> None:
        weekday = ts.dt.dayofweek

        # Saturday candles are always wrong
        saturday_mask = weekday == _SATURDAY
        n_sat = int(saturday_mask.sum())

        # Sunday candles before 21:00 UTC are suspect (market closed)
        sunday_early = (weekday == _SUNDAY) & (ts.dt.hour < 21)
        n_sun = int(sunday_early.sum())

        report.weekend_candles = n_sat + n_sun
        if report.weekend_candles:
            report.warnings.append(
                f"{n_sat} Saturday + {n_sun} early-Sunday candles found "
                "(market closed; likely synthetic filler data)."
            )

    @staticmethod
    def _check_thin_market_candles(ts: pd.Series, report: CalendarReport) -> None:
        mask = ts.apply(
            lambda t: (t.month, t.day) in _THIN_MARKET_DATES
        )
        report.thin_market_rows = int(mask.sum())
        if report.thin_market_rows:
            report.warnings.append(
                f"{report.thin_market_rows} rows on known thin-market dates "
                "(Christmas / New Year). Spreads are typically wider."
            )

    @staticmethod
    def _classify_gaps(
        ts: pd.Series,
        timeframe: str,
        report: CalendarReport,
    ) -> None:
        expected  = _FREQ_MAP[timeframe]
        threshold = expected * _INTRA_WEEK_MULTIPLIER

        sorted_ts = ts.sort_values().reset_index(drop=True)
        diffs = sorted_ts.diff().dropna()

        for idx, gap in diffs.items():
            if gap <= threshold:
                continue

            prev_ts = sorted_ts.iloc[idx - 1]
            curr_ts = sorted_ts.iloc[idx]

            # Classify as weekend gap (Fri→Mon) or unexpected
            prev_wd = prev_ts.dayofweek
            curr_wd = curr_ts.dayofweek

            is_weekend_gap = (
                prev_wd in (4, _SATURDAY, _SUNDAY) and   # Friday or later
                curr_wd in (0, 1) and                     # Monday or Tuesday
                _WEEKEND_MIN_GAP <= gap <= _WEEKEND_MAX_GAP
            )

            is_holiday_gap = (
                (prev_ts.month, prev_ts.day) in _THIN_MARKET_DATES or
                (curr_ts.month, curr_ts.day) in _THIN_MARKET_DATES
            )

            detail = {
                "from": str(prev_ts),
                "to":   str(curr_ts),
                "gap":  str(gap),
                "type": "weekend" if is_weekend_gap else ("holiday" if is_holiday_gap else "unexpected"),
            }
            report.gap_details.append(detail)

            if is_weekend_gap or is_holiday_gap:
                report.expected_gaps += 1
            else:
                report.unexpected_gaps += 1

        if report.unexpected_gaps:
            report.warnings.append(
                f"{report.unexpected_gaps} unexpected intra-week gaps > {threshold}. "
                "Check broker connectivity or missing data around these periods."
            )
