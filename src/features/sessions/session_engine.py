"""Sessions Engine — LuxAlgo Sessions + ICT Killzones combined.

Converts all four major forex sessions and the two primary ICT Kill Zones into
25 ML-ready float64 features.  Every visual Pine Script object (range boxes,
trendlines, background colours, labels) is replaced by a numerical equivalent.

Session UTC hours (standard forex market-hours, DST-agnostic at the instrument
level — adjust with tz_offset_hours if your broker uses a shifted feed):

    Sydney   21:00 – 06:00  (crosses UTC midnight)
    Asia     00:00 – 09:00
    London   07:00 – 16:00
    New York 13:00 – 22:00

ICT Kill Zone UTC hours:
    London Kill Zone    02:00 – 05:00
    New York Kill Zone  07:00 – 10:00

Dominant-session priority when sessions overlap (highest = wins stats):
    New York (4) > London (3) > Asia (2) > Sydney (1) > None (0)

session_id encodes simultaneous memberships as a bitfield:
    bit 0 (1) = Sydney, bit 1 (2) = Asia,
    bit 2 (4) = London, bit 3 (8) = New York
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

# ─── Session constants ────────────────────────────────────────────────────────
_SESS_NONE   = 0.0
_SESS_SYDNEY = 1.0
_SESS_ASIA   = 2.0
_SESS_LONDON = 3.0
_SESS_NY     = 4.0

# Session open/close in UTC hours (integer, exclusive end)
_SYDNEY_OPEN  = 21;  _SYDNEY_CLOSE  =  6   # crosses midnight
_ASIA_OPEN    =  0;  _ASIA_CLOSE    =  9
_LONDON_OPEN  =  7;  _LONDON_CLOSE  = 16
_NY_OPEN      = 13;  _NY_CLOSE      = 22

# ICT Kill Zone hours
_LK_OPEN  =  2;  _LK_CLOSE  =  5   # London Kill Zone
_NK_OPEN  =  7;  _NK_CLOSE  = 10   # New York Kill Zone

_SESSION_CLOSE_MINUTES: dict[float, int] = {
    _SESS_SYDNEY: _SYDNEY_CLOSE * 60,   # 360
    _SESS_ASIA:   _ASIA_CLOSE   * 60,   # 540
    _SESS_LONDON: _LONDON_CLOSE * 60,   # 960
    _SESS_NY:     _NY_CLOSE     * 60,   # 1320
}

_OUTPUT_COLUMNS: list[str] = [
    "session",
    "session_id",
    "is_london",
    "is_new_york",
    "is_asia",
    "is_sydney",
    "is_london_killzone",
    "is_newyork_killzone",
    "session_overlap",
    "session_high",
    "session_low",
    "session_mid",
    "session_range",
    "session_vwap",
    "session_mean",
    "session_volatility",
    "session_momentum",
    "session_volume",
    "session_delta",
    "session_trend",
    "session_liquidity",
    "minutes_since_session_open",
    "minutes_until_session_close",
    "opening_range_breakout",
    "adr_position",
]


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _dominant_groups(dom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign a monotonically increasing group ID per dominant-session run.

    A new group starts whenever `dom` changes value *and* the new value is > 0.
    Out-of-session bars receive group 0.

    Returns
    -------
    (group_ids, is_in_session)
    """
    is_in   = dom > 0.0
    # Detect every change in the dominant session (including 0 → session and
    # session → 0 transitions).
    changes = np.empty(len(dom), dtype=bool)
    changes[0]  = is_in[0]                     # first in-session bar starts group 1
    changes[1:] = dom[1:] != dom[:-1]
    new_grp = changes & is_in
    grp_num = np.cumsum(new_grp).astype(np.intp)
    return np.where(is_in, grp_num, 0), is_in


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_c = np.empty_like(close)
    prev_c[0]  = close[0]
    prev_c[1:] = close[:-1]
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)),
    )


def _infer_bar_minutes(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 15.0
    return max(1.0, (df.index[1] - df.index[0]).total_seconds() / 60.0)


def _minutes_until_close(
    h: np.ndarray, m: np.ndarray, dom: np.ndarray
) -> np.ndarray:
    """Minutes remaining until the dominant session closes (0 if not in session)."""
    msm = h * 60.0 + m   # minutes since midnight

    asia_u   = np.clip(_ASIA_CLOSE   * 60 - msm, 0.0, None)
    london_u = np.clip(_LONDON_CLOSE * 60 - msm, 0.0, None)
    ny_u     = np.clip(_NY_CLOSE     * 60 - msm, 0.0, None)

    # Sydney crosses midnight: evening part (h ≥ 21) wraps to next day
    sydney_u = np.where(
        h >= _SYDNEY_OPEN,
        (24.0 - h) * 60.0 - m + _SYDNEY_CLOSE * 60.0,
        np.clip(_SYDNEY_CLOSE * 60.0 - msm, 0.0, None),
    )

    return np.select(
        [dom == _SESS_SYDNEY, dom == _SESS_ASIA,
         dom == _SESS_LONDON, dom == _SESS_NY],
        [sydney_u, asia_u, london_u, ny_u],
        default=0.0,
    )


# ─── Engine ───────────────────────────────────────────────────────────────────

@FeatureRegistry.register
class SessionEngine(BaseFeature):
    """LuxAlgo Sessions + ICT Kill Zones combined into 25 ML-ready features.

    All statistics (high, low, VWAP, delta, momentum …) track the dominant
    session only and reset when a new session instance begins.  Out-of-session
    bars receive neutral values so no NaN reaches downstream models.
    """

    name:             str       = "sessions"
    category:         str       = "sessions"
    dependencies:     list[str] = []
    required_columns: list[str] = ["open", "high", "low", "close", "volume"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0914, PLR0915
        idx = df.index

        # ── UTC time arrays ────────────────────────────────────────────────────
        if hasattr(idx, "tz") and idx.tz is not None:
            utc_idx = idx.tz_convert("UTC")
        else:
            logger.warning(
                "[sessions] Index has no timezone — assuming UTC. "
                "Attach UTC timezone for correct session detection."
            )
            utc_idx = idx

        h = np.asarray(utc_idx.hour,   dtype=np.float64)
        m = np.asarray(utc_idx.minute, dtype=np.float64)

        open_arr  = df["open"].to_numpy(dtype=np.float64)
        high_arr  = df["high"].to_numpy(dtype=np.float64)
        low_arr   = df["low"].to_numpy(dtype=np.float64)
        close_arr = df["close"].to_numpy(dtype=np.float64)
        vol_arr   = df["volume"].to_numpy(dtype=np.float64)

        # ── Session membership (vectorised) ────────────────────────────────────
        is_sydney = (h >= _SYDNEY_OPEN) | (h < _SYDNEY_CLOSE)
        is_asia   = (h >= _ASIA_OPEN)   & (h < _ASIA_CLOSE)
        is_london = (h >= _LONDON_OPEN) & (h < _LONDON_CLOSE)
        is_ny     = (h >= _NY_OPEN)     & (h < _NY_CLOSE)

        # ── Kill-zone flags ────────────────────────────────────────────────────
        is_lk = (h >= _LK_OPEN) & (h < _LK_CLOSE)
        is_nk = (h >= _NK_OPEN) & (h < _NK_CLOSE)

        # ── session_id bitfield (simultaneous memberships) ────────────────────
        session_id = (
            is_sydney.astype(np.float64) * 1.0
            + is_asia.astype(np.float64)   * 2.0
            + is_london.astype(np.float64) * 4.0
            + is_ny.astype(np.float64)     * 8.0
        )

        # ── Dominant session (priority: NY > London > Asia > Sydney) ──────────
        dom = np.full(len(df), _SESS_NONE, dtype=np.float64)
        dom = np.where(is_sydney, _SESS_SYDNEY, dom)
        dom = np.where(is_asia,   _SESS_ASIA,   dom)
        dom = np.where(is_london, _SESS_LONDON, dom)
        dom = np.where(is_ny,     _SESS_NY,     dom)

        # ── Session overlap: two or more sessions simultaneously active ────────
        overlap = (
            (is_london & is_ny)
            | (is_london & is_asia)
            | (is_sydney & is_asia)
        )

        # ── Group IDs for the dominant session ────────────────────────────────
        dom_groups, is_in = _dominant_groups(dom)
        grp_s = pd.Series(dom_groups, index=idx, name="grp")

        # ── Build working frame for groupby operations ─────────────────────────
        tp   = (high_arr + low_arr + close_arr) / 3.0
        tr   = _true_range(high_arr, low_arr, close_arr)
        bull = np.where(close_arr > open_arr, vol_arr, 0.0)
        bear = np.where(close_arr < open_arr, vol_arr, 0.0)
        dv   = bull - bear   # bar-level directional volume

        wk = pd.DataFrame(
            {
                "high":  high_arr,
                "low":   low_arr,
                "close": close_arr,
                "vol":   vol_arr,
                "tpv":   tp * vol_arr,
                "tr":    tr,
                "dv":    dv,
            },
            index=idx,
        )

        # ── Cumulative session statistics ──────────────────────────────────────
        g = wk.groupby(grp_s)

        sess_high_s  = g["high"].cummax()
        sess_low_s   = g["low"].cummin()
        sess_vol_s   = g["vol"].cumsum()
        sess_delta_s = g["dv"].cumsum()
        cum_tpv_s    = g["tpv"].cumsum()
        cum_vol_s    = g["vol"].cumsum()
        cum_c_s      = g["close"].cumsum()
        cum_tr_s     = g["tr"].cumsum()
        bar_cnt_s    = g.cumcount() + 1          # 1 for first bar in group
        open_c_s     = g["close"].transform("first")   # session open price
        open_h_s     = g["high"].transform("first")    # opening-bar high
        open_l_s     = g["low"].transform("first")     # opening-bar low

        # Convert to numpy
        sess_high  = sess_high_s.to_numpy()
        sess_low   = sess_low_s.to_numpy()
        sess_vol   = sess_vol_s.to_numpy()
        sess_delta = sess_delta_s.to_numpy()
        cum_tpv    = cum_tpv_s.to_numpy()
        cum_vol    = cum_vol_s.to_numpy()
        bar_cnt    = bar_cnt_s.to_numpy()
        sess_open  = open_c_s.to_numpy()
        open_h     = open_h_s.to_numpy()
        open_l     = open_l_s.to_numpy()

        safe_cv = np.where(cum_vol > 0, cum_vol, 1e-8)
        sess_vwap    = cum_tpv / safe_cv
        sess_mean    = cum_c_s.to_numpy() / bar_cnt
        sess_volatil = cum_tr_s.to_numpy() / bar_cnt  # mean ATR within session

        # ── Mask out-of-session bars with neutral values ───────────────────────
        sess_high    = np.where(is_in, sess_high,    close_arr)
        sess_low     = np.where(is_in, sess_low,     close_arr)
        sess_vol     = np.where(is_in, sess_vol,     0.0)
        sess_delta   = np.where(is_in, sess_delta,   0.0)
        sess_vwap    = np.where(is_in, sess_vwap,    close_arr)
        sess_mean    = np.where(is_in, sess_mean,    close_arr)
        sess_volatil = np.where(is_in, sess_volatil, 0.0)

        sess_range = sess_high - sess_low
        sess_mid   = (sess_high + sess_low) * 0.5

        # ── Momentum, trend, liquidity ─────────────────────────────────────────
        safe_open = np.where((is_in) & (sess_open > 0), sess_open, 1.0)
        sess_momentum = np.where(
            is_in,
            (close_arr - sess_open) / safe_open * 100.0,
            0.0,
        )
        sess_trend = np.sign(sess_momentum)

        sess_liq = np.where(
            is_in & (sess_range > 0),
            sess_vol / (sess_range + 1e-8),
            0.0,
        )

        # ── Time metrics ───────────────────────────────────────────────────────
        bar_min   = _infer_bar_minutes(df)
        mins_since = np.where(
            is_in,
            (bar_cnt - 1) * bar_min,   # 0 on the opening bar
            0.0,
        )
        mins_until = _minutes_until_close(h, m, dom)

        # ── Opening range breakout ─────────────────────────────────────────────
        orb = np.where(
            is_in,
            np.where(
                close_arr > open_h, 1.0,
                np.where(close_arr < open_l, -1.0, 0.0),
            ),
            0.0,
        )

        # ── ADR position (within today's cumulative daily range) ──────────────
        date_s  = pd.Series(utc_idx.date, index=idx, name="date")
        day_h_s = pd.Series(high_arr,  index=idx).groupby(date_s).cummax()
        day_l_s = pd.Series(low_arr,   index=idx).groupby(date_s).cummin()
        day_h   = day_h_s.to_numpy()
        day_l   = day_l_s.to_numpy()
        day_rng = day_h - day_l
        adr_pos = np.where(
            day_rng > 0,
            np.clip((close_arr - day_l) / day_rng, 0.0, 1.0),
            0.5,
        )

        # ── Assemble output ────────────────────────────────────────────────────
        out = pd.DataFrame(index=idx)
        out["session"]                    = dom
        out["session_id"]                 = session_id
        out["is_london"]                  = is_london.astype(np.float64)
        out["is_new_york"]                = is_ny.astype(np.float64)
        out["is_asia"]                    = is_asia.astype(np.float64)
        out["is_sydney"]                  = is_sydney.astype(np.float64)
        out["is_london_killzone"]         = is_lk.astype(np.float64)
        out["is_newyork_killzone"]        = is_nk.astype(np.float64)
        out["session_overlap"]            = overlap.astype(np.float64)
        out["session_high"]               = sess_high
        out["session_low"]                = sess_low
        out["session_mid"]                = sess_mid
        out["session_range"]              = sess_range
        out["session_vwap"]               = sess_vwap
        out["session_mean"]               = sess_mean
        out["session_volatility"]         = sess_volatil
        out["session_momentum"]           = sess_momentum
        out["session_volume"]             = sess_vol
        out["session_delta"]              = sess_delta
        out["session_trend"]              = sess_trend
        out["session_liquidity"]          = sess_liq
        out["minutes_since_session_open"] = mins_since
        out["minutes_until_session_close"]= mins_until
        out["opening_range_breakout"]     = orb
        out["adr_position"]               = adr_pos

        return out.astype(np.float64)

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "LuxAlgo Sessions + ICT Kill Zones.  Detects all four major "
                "forex sessions (Sydney, Asia, London, New York) and the two "
                "primary ICT Kill Zones, then computes 25 ML-ready features: "
                "session flags, running H/L/VWAP/volume, momentum, trend, "
                "time-to-close, opening-range breakout, and ADR position."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = _OUTPUT_COLUMNS,
            version          = "1.0.0",
            author           = "Smart Trading Team",
            complexity       = "low",
            tags             = [
                "ICT", "LuxAlgo", "sessions", "killzones",
                "london", "new_york", "asia", "sydney",
                "vwap", "opening_range", "session_structure",
            ],
        )
