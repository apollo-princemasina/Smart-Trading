"""Liquidity Sweep Engine — HexaTrades Advanced Liquidity Sweep logic.

Tracks buy-side and sell-side liquidity levels (swing highs/lows, EQH/EQL)
and detects when price sweeps (takes out) those levels.

Design
------
Reads all pivot, swing, and EQH/EQL columns from the enriched running_df
produced by MarketStructureEngine and EqualHighsLowsEngine.  No pivots are
recalculated here.

Performance
-----------
Active levels are stored in fixed-size numpy arrays (_MAX_LEVELS = 20 slots).
Sweep detection, score computation, and nearest-level queries are all
vectorized over those 20 slots, so the outer Python loop (87 K bars) does
only O(_MAX_LEVELS) numpy work per iteration.

Liquidity Level Representation
--------------------------------
Each of the K = _MAX_LEVELS slots holds:
  lev_price[k]    : pivot or EQH/EQL price
  lev_bar[k]      : row offset (int) when the level was formed
  lev_is_high[k]  : True = buy-side (above price), False = sell-side (below)
  lev_is_major[k] : True = major 15-bar pivot, False = minor 5-bar
  lev_is_equal[k] : part of an EQH/EQL cluster
  lev_touches[k]  : cluster strength (number of equal touches)
  lev_active[k]   : True = slot occupied and not yet swept

Sweep Detection
---------------
  Bullish sweep (sell-side taken):  low[i] < lev_price AND close[i] > lev_price
  Bearish sweep (buy-side taken):   high[i] > lev_price AND close[i] < lev_price

Score Weighting (Balanced preset)
-----------------------------------
  freshness  = max(0, 1 - age/200) * 20
  cluster    = min(touches, 5) / 5 * 30
  structural = 25 if major else 10
  equal_prem = 15 if equal else 0
  bos_conf   = 10 if BOS/CHoCH at formation else 0
  score      = min(100, sum of above)

Output Columns (18 float64)
----------------------------
bullish_liquidity_sweep    : 1.0 when a sell-side level is swept bullishly
bearish_liquidity_sweep    : 1.0 when a buy-side level is swept bearishly
liquidity_score            : 0-100 composite strength of nearest unswept level
nearest_liquidity_distance : % distance from close to nearest unswept level
nearest_buy_liquidity      : price of nearest unswept buy-side level
nearest_sell_liquidity     : price of nearest unswept sell-side level
liquidity_age              : bars since nearest level was formed
touch_count                : touches (cluster size) of nearest level
strong_sweep               : 1.0 if sweep score >= strongScore threshold (70)
weak_sweep                 : 1.0 if sweep score < threshold on sweep bar
confirmed_sweep            : 1.0 on bar AFTER sweep when next candle confirms
sweep_strength             : sweep penetration / ATR (normalised)
liquidity_cluster_size     : equal-level touches of nearest level
sweep_penetration          : how far price moved past the level (% of price)
sweep_rejection            : wick back from sweep extreme toward close (%)
liq_zone_width             : EQH/EQL tolerance band width (% of price)
liq_zone_lifetime          : bars the nearest active zone has existed
num_nearby_liq_pools       : count of unswept levels within proximity_pct
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

# ── Tuneable constants ────────────────────────────────────────────────────────
_MAX_LEVELS:    int   = 20      # hard cap on tracked levels (Pine Script default)
_STRONG_SCORE:  float = 70.0   # minimum score for a "strong" sweep
_ATR_PERIOD:    int   = 14     # ATR window for sweep normalisation
_EQUAL_PCT:     float = 0.0005 # 0.05% tolerance for equal-level zone width
_PROXIMITY_PCT: float = 0.005  # 0.50% band for counting nearby pools

# ── Score weights (Balanced preset) ──────────────────────────────────────────
_W_AGE      = 20.0
_W_CLUSTER  = 30.0
_W_MAJOR    = 25.0
_W_MINOR    = 10.0
_W_EQUAL    = 15.0
_W_BOS      = 10.0
_AGE_DECAY  = 200.0   # bars over which freshness decays to zero

K = _MAX_LEVELS       # slot dimension alias


@FeatureRegistry.register
class LiquiditySweepEngine(BaseFeature):
    """Detect liquidity sweeps and compute composite sweep/pool features.

    Reads columns from the accumulated running_df; never recalculates pivots.
    """

    name:             str       = "liquidity_sweeps"
    category:         str       = "liquidity"
    dependencies:     list[str] = ["market_structure", "bos_choch", "equal_highs_lows"]
    required_columns: list[str] = [
        # OHLCV
        "open", "high", "low", "close",
        # Pivots
        "pivot_high", "pivot_low",
        "major_pivot_high", "major_pivot_low",
        # Swing prices (ffill)
        "swing_high_price", "swing_low_price",
        # BOS/CHoCH confluence flags
        "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
        # Equal Highs/Lows
        "eqh", "eql", "eqh_price", "eql_price", "eqh_age", "eql_age",
    ]

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0914
        n   = len(df)
        idx = df.index

        if n == 0:
            return pd.DataFrame(
                columns=[
                    "bullish_liquidity_sweep", "bearish_liquidity_sweep",
                    "liquidity_score", "nearest_liquidity_distance",
                    "nearest_buy_liquidity", "nearest_sell_liquidity",
                    "liquidity_age", "touch_count",
                    "strong_sweep", "weak_sweep", "confirmed_sweep",
                    "sweep_strength", "liquidity_cluster_size",
                    "sweep_penetration", "sweep_rejection",
                    "liq_zone_width", "liq_zone_lifetime", "num_nearby_liq_pools",
                ],
                index=idx,
                dtype=float,
            )

        # ── Pre-compute ATR ───────────────────────────────────────────────────
        atr_arr = self._compute_atr(
            df["high"], df["low"], df["close"], _ATR_PERIOD
        ).to_numpy(dtype=float)

        # ── Extract source arrays ─────────────────────────────────────────────
        high_arr  = df["high"].to_numpy(dtype=float)
        low_arr   = df["low"].to_numpy(dtype=float)
        close_arr = df["close"].to_numpy(dtype=float)

        pivot_high_arr  = df["pivot_high"].to_numpy(dtype=float)
        pivot_low_arr   = df["pivot_low"].to_numpy(dtype=float)
        major_ph_arr    = df["major_pivot_high"].to_numpy(dtype=float)
        major_pl_arr    = df["major_pivot_low"].to_numpy(dtype=float)
        eqh_arr         = df["eqh"].to_numpy(dtype=float)
        eql_arr         = df["eql"].to_numpy(dtype=float)
        bos_bull_arr    = df["bos_bullish"].to_numpy(dtype=float)
        bos_bear_arr    = df["bos_bearish"].to_numpy(dtype=float)
        choch_bull_arr  = df["choch_bullish"].to_numpy(dtype=float)
        choch_bear_arr  = df["choch_bearish"].to_numpy(dtype=float)

        # ── Fixed-size level slot arrays ──────────────────────────────────────
        lev_price    = np.zeros(K, dtype=float)
        lev_bar      = np.full(K, -1, dtype=np.int64)
        lev_is_high  = np.zeros(K, dtype=bool)
        lev_is_major = np.zeros(K, dtype=bool)
        lev_is_equal = np.zeros(K, dtype=bool)
        lev_touches  = np.ones(K, dtype=float)
        lev_active   = np.zeros(K, dtype=bool)   # True = occupied & not swept
        lev_bos_conf = np.zeros(K, dtype=bool)   # BOS/CHoCH at formation bar

        # ── Output arrays ─────────────────────────────────────────────────────
        bull_sweep      = np.zeros(n, dtype=float)
        bear_sweep      = np.zeros(n, dtype=float)
        liq_score       = np.zeros(n, dtype=float)
        nearest_dist    = np.full(n, np.nan, dtype=float)
        nearest_buy     = np.full(n, np.nan, dtype=float)
        nearest_sell    = np.full(n, np.nan, dtype=float)
        liq_age_out     = np.zeros(n, dtype=float)
        touch_out       = np.zeros(n, dtype=float)
        strong_out      = np.zeros(n, dtype=float)
        weak_out        = np.zeros(n, dtype=float)
        confirmed_out   = np.zeros(n, dtype=float)
        sweep_str_out   = np.zeros(n, dtype=float)
        cluster_out     = np.zeros(n, dtype=float)
        pen_out         = np.zeros(n, dtype=float)
        rej_out         = np.zeros(n, dtype=float)
        zwidth_out      = np.zeros(n, dtype=float)
        zlife_out       = np.zeros(n, dtype=float)
        pools_out       = np.zeros(n, dtype=float)

        # previous-bar sweep state for confirmed_sweep
        prev_bull_sweep = False
        prev_bear_sweep = False
        prev_close      = np.nan

        for i in range(n):
            c     = close_arr[i]
            h_i   = high_arr[i]
            lo_i  = low_arr[i]
            atr_i = atr_arr[i] if not np.isnan(atr_arr[i]) and atr_arr[i] > 0 else 1e-8
            bos_c = (bos_bull_arr[i] == 1.0 or bos_bear_arr[i] == 1.0 or
                     choch_bull_arr[i] == 1.0 or choch_bear_arr[i] == 1.0)

            # ── 1. Confirmed sweep from prior bar ─────────────────────────────
            if i > 0:
                if prev_bull_sweep and c > prev_close:
                    confirmed_out[i] = 1.0
                elif prev_bear_sweep and c < prev_close:
                    confirmed_out[i] = 1.0

            # ── 2. Ingest new pivot levels ────────────────────────────────────
            if pivot_high_arr[i] == 1.0:
                self._add_level(
                    lev_price, lev_bar, lev_is_high, lev_is_major,
                    lev_is_equal, lev_touches, lev_active, lev_bos_conf,
                    price    = h_i,
                    bar      = i,
                    is_high  = True,
                    is_major = major_ph_arr[i] == 1.0,
                    is_equal = eqh_arr[i] == 1.0,
                    touches  = 2.0 if eqh_arr[i] == 1.0 else 1.0,
                    bos_conf = bos_c,
                )

            if pivot_low_arr[i] == 1.0:
                self._add_level(
                    lev_price, lev_bar, lev_is_high, lev_is_major,
                    lev_is_equal, lev_touches, lev_active, lev_bos_conf,
                    price    = lo_i,
                    bar      = i,
                    is_high  = False,
                    is_major = major_pl_arr[i] == 1.0,
                    is_equal = eql_arr[i] == 1.0,
                    touches  = 2.0 if eql_arr[i] == 1.0 else 1.0,
                    bos_conf = bos_c,
                )

            # ── 3. Vectorized sweep detection over all active slots ────────────
            act = lev_active                           # boolean mask, length K

            if act.any():
                prices = lev_price                     # (K,)

                # Bearish sweep: buy-side level (is_high) pierced above, close below
                bear_mask = act & lev_is_high & (h_i > prices) & (c < prices)
                # Bullish sweep: sell-side level (not is_high) pierced below, close above
                bull_mask = act & ~lev_is_high & (lo_i < prices) & (c > prices)

                # Compute scores for active levels (vectorized)
                ages    = np.where(act, i - lev_bar, 0).astype(float)
                scores  = self._score_vec(
                    ages, lev_touches, lev_is_major, lev_is_equal, lev_bos_conf
                )

                if bull_mask.any():
                    bull_sweep[i] = 1.0
                    best_k = int(np.argmax(np.where(bull_mask, scores, -1.0)))
                    dom_score = scores[best_k]
                    lv_price  = prices[best_k]
                    strong_out[i]   = 1.0 if dom_score >= _STRONG_SCORE else 0.0
                    weak_out[i]     = 0.0 if dom_score >= _STRONG_SCORE else 1.0
                    pen_pct         = (lv_price - lo_i) / (lv_price + 1e-12) * 100.0
                    rej_pct         = (c - lo_i)        / (lv_price + 1e-12) * 100.0
                    pen_out[i]      = pen_pct
                    rej_out[i]      = rej_pct
                    sweep_str_out[i] = pen_pct / (atr_i / (c + 1e-12) * 100.0 + 1e-12)
                    lev_active[bull_mask] = False

                if bear_mask.any():
                    bear_sweep[i] = 1.0
                    best_k = int(np.argmax(np.where(bear_mask, scores, -1.0)))
                    dom_score = scores[best_k]
                    lv_price  = prices[best_k]
                    if bull_sweep[i] == 0.0:   # don't overwrite if both fired
                        strong_out[i]   = 1.0 if dom_score >= _STRONG_SCORE else 0.0
                        weak_out[i]     = 0.0 if dom_score >= _STRONG_SCORE else 1.0
                    pen_pct          = (h_i - lv_price) / (lv_price + 1e-12) * 100.0
                    rej_pct          = (h_i - c)        / (lv_price + 1e-12) * 100.0
                    pen_out[i]       = max(pen_out[i], pen_pct)
                    rej_out[i]       = max(rej_out[i], rej_pct)
                    if sweep_str_out[i] == 0.0:
                        sweep_str_out[i] = pen_pct / (atr_i / (c + 1e-12) * 100.0 + 1e-12)
                    lev_active[bear_mask] = False

                # ── 4. Nearest-level stats (remaining active) ─────────────────
                act2 = lev_active
                if act2.any():
                    dists = np.where(act2, np.abs(c - lev_price) / (c + 1e-12) * 100.0, np.inf)
                    nearest_idx   = int(np.argmin(dists))
                    nearest_dist[i]  = dists[nearest_idx]
                    liq_score[i]     = scores[nearest_idx]   # reuse scores from pre-sweep
                    liq_age_out[i]   = float(i - lev_bar[nearest_idx])
                    touch_out[i]     = lev_touches[nearest_idx]
                    cluster_out[i]   = lev_touches[nearest_idx]
                    zwidth_out[i]    = _EQUAL_PCT * lev_price[nearest_idx] * 100.0
                    zlife_out[i]     = float(i - lev_bar[nearest_idx])

                    # Nearest buy-side
                    buy_dists  = np.where(act2 & lev_is_high,  dists, np.inf)
                    sell_dists = np.where(act2 & ~lev_is_high, dists, np.inf)
                    if np.isfinite(buy_dists).any():
                        nearest_buy[i] = lev_price[int(np.argmin(buy_dists))]
                    if np.isfinite(sell_dists).any():
                        nearest_sell[i] = lev_price[int(np.argmin(sell_dists))]

                    # Pool count within proximity band
                    pools_out[i] = float(
                        np.sum(act2 & (dists <= _PROXIMITY_PCT * 100.0))
                    )

            # ── 5. Advance previous-bar state ─────────────────────────────────
            prev_bull_sweep = bull_sweep[i] == 1.0
            prev_bear_sweep = bear_sweep[i] == 1.0
            prev_close      = c

        # ── Assemble output DataFrame ─────────────────────────────────────────
        return pd.DataFrame(
            {
                "bullish_liquidity_sweep":    bull_sweep,
                "bearish_liquidity_sweep":    bear_sweep,
                "liquidity_score":            liq_score,
                "nearest_liquidity_distance": nearest_dist,
                "nearest_buy_liquidity":      nearest_buy,
                "nearest_sell_liquidity":     nearest_sell,
                "liquidity_age":              liq_age_out,
                "touch_count":               touch_out,
                "strong_sweep":              strong_out,
                "weak_sweep":                weak_out,
                "confirmed_sweep":           confirmed_out,
                "sweep_strength":            sweep_str_out,
                "liquidity_cluster_size":    cluster_out,
                "sweep_penetration":         pen_out,
                "sweep_rejection":           rej_out,
                "liq_zone_width":            zwidth_out,
                "liq_zone_lifetime":         zlife_out,
                "num_nearby_liq_pools":      pools_out,
            },
            index=idx,
        )

    # ── Slot management ───────────────────────────────────────────────────────

    @staticmethod
    def _add_level(
        lev_price, lev_bar, lev_is_high, lev_is_major,
        lev_is_equal, lev_touches, lev_active, lev_bos_conf,
        price, bar, is_high, is_major, is_equal, touches, bos_conf,
    ) -> None:
        """Insert a new level into the first empty slot, evicting the oldest if full."""
        # Find an empty slot
        empty = np.where(~lev_active)[0]
        if len(empty) > 0:
            k = int(empty[0])
        else:
            # All slots full: evict the oldest active level
            k = int(np.argmin(lev_bar))

        lev_price[k]    = price
        lev_bar[k]      = bar
        lev_is_high[k]  = is_high
        lev_is_major[k] = is_major
        lev_is_equal[k] = is_equal
        lev_touches[k]  = touches
        lev_active[k]   = True
        lev_bos_conf[k] = bos_conf

    # ── Vectorized scoring ────────────────────────────────────────────────────

    @staticmethod
    def _score_vec(
        ages:     np.ndarray,   # (K,) float — bars since formation
        touches:  np.ndarray,   # (K,) float
        is_major: np.ndarray,   # (K,) bool
        is_equal: np.ndarray,   # (K,) bool
        bos_conf: np.ndarray,   # (K,) bool
    ) -> np.ndarray:
        """Vectorized 0-100 composite score for all K level slots."""
        freshness  = np.clip(1.0 - ages / _AGE_DECAY, 0.0, 1.0) * _W_AGE
        cluster    = np.clip(touches / 5.0, 0.0, 1.0) * _W_CLUSTER
        structural = np.where(is_major, _W_MAJOR, _W_MINOR)
        equal_prem = np.where(is_equal, _W_EQUAL, 0.0)
        bos_prem   = np.where(bos_conf, _W_BOS,   0.0)
        return np.clip(freshness + cluster + structural + equal_prem + bos_prem, 0.0, 100.0)

    # ── ATR helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int,
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    # ── Metadata ──────────────────────────────────────────────────────────────

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "HexaTrades-style Liquidity Sweep Engine.  Tracks buy-side and "
                "sell-side liquidity pools (pivot highs/lows, EQH/EQL clusters) "
                "and detects when price sweeps through them.  Outputs composite "
                "sweep strength, nearest-pool distances, and ML-ready binary "
                "sweep flags."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = [
                "bullish_liquidity_sweep", "bearish_liquidity_sweep",
                "liquidity_score", "nearest_liquidity_distance",
                "nearest_buy_liquidity", "nearest_sell_liquidity",
                "liquidity_age", "touch_count",
                "strong_sweep", "weak_sweep", "confirmed_sweep",
                "sweep_strength", "liquidity_cluster_size",
                "sweep_penetration", "sweep_rejection",
                "liq_zone_width", "liq_zone_lifetime", "num_nearby_liq_pools",
            ],
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "high",
            tags       = [
                "ICT", "smart_money", "liquidity",
                "sweep", "stop_hunt", "buy_side", "sell_side",
                "EQH", "EQL", "liquidity_pool",
            ],
        )
