"""Liquidity Magnet Engine — translated from LunqFX Liquidity Magnet Pine Script.

Every resting liquidity pool (swing high / low, EQH / EQL cluster) is scored
0-100 by how strongly it is likely to be swept next:

    Magnet Score = proximity + age + touches + momentum  (weighted, max = 100)

Weights (Balanced / Pine Script defaults)
-----------------------------------------
    Proximity   55 pts  — closer to price = higher pull
    Momentum    25 pts  — price already moving toward the level
    Age         10 pts  — fresher pools score higher
    Touches     10 pts  — more equal-level touches = stronger pool

The highest-scoring pool within ``maxDist`` % (5 %) of the current close and
with a score >= ``minScore`` (35) becomes the NEXT TARGET.  All outputs are
numerical ML features — no plotting, no labels, no TradingView objects.

Non-Repainting
--------------
Pools form only on *confirmed* pivots (``pivot_high`` / ``pivot_low`` from
MarketStructureEngine, which require right-side confirmation).  Sweep detection
fires only when the *close* crosses through the level, never intra-bar.

Design: What This Module Reuses
--------------------------------
From MarketStructureEngine
    ``pivot_high``, ``pivot_low``             — pool formation triggers
    ``major_pivot_high``, ``major_pivot_low`` — structural tier flag

From EqualHighsLowsEngine
    ``eqh``, ``eql``                          — initial touch-count hint
    (touch merging re-uses ATR-based tolerance independently for the magnet's
    own ``eqTol × ATR`` threshold, which differs from the 0.05 % fixed band)

The liquidity_sweeps dependency guarantees pipeline ordering only; no columns
from that engine are consumed here — the magnet maintains its own pool set
with its own scoring formula (momentum-weighted, 7 pools/side maximum).

Pool Management
---------------
Up to K_SIDE (= 7) pools per side are tracked in fixed-size numpy slot arrays.
    • New pivot arrives within ``eqTol × ATR`` of an existing pool →
      touch count incremented, no new slot opened.
    • All K_SIDE slots full → oldest slot is evicted.
    • Pool age > ``ageMax`` (400 bars) → age-out removal.
    • Close crosses through pool price (sweep condition) → pool removed.

Performance
-----------
Two sets of K_SIDE (= 7) numpy slot arrays replace Python dicts.  All scoring,
sweep detection, and nearest-level queries are vectorised over 14 total slots.
The outer Python loop touches 87 K bars only; per-bar numpy work is O(14).

Output Columns (20, all float64)
----------------------------------
nearest_buy_liquidity_distance   : % distance from close to nearest active buy-side pool
nearest_sell_liquidity_distance  : % distance from close to nearest active sell-side pool
nearest_liquidity_score          : magnet score of the nearest active pool (any side)
magnet_score                     : magnet score of the NEXT TARGET pool (0 if no target)
magnet_probability               : magnet_score / 100 → [0, 1]
liquidity_rank                   : rank of target by magnet score among all active pools
                                   (1 = strongest pull in the market, 0 = no target)
target_liquidity                 : price of the NEXT TARGET pool (0.0 if no target)
distance_to_target               : % distance from close to target (0.0 if no target)
buy_side_probability             : score-weighted fraction of pull toward buy-side levels
sell_side_probability            : score-weighted fraction of pull toward sell-side levels
liquidity_density                : count of active pools within 2 × ATR of close
cluster_strength                 : touch count of the NEXT TARGET pool
magnet_strength                  : max magnet score across ALL active pools
nearest_cluster_size             : touch count of the nearest (by distance) pool
proximity_contribution           : proximity component of target score  [0, 55]
age_contribution                 : age component of target score        [0, 10]
touch_contribution               : touch component of target score      [0, 10]
momentum_contribution            : momentum component of target score   [0, 25]
ranking_position                 : magnet-score rank of the nearest pool [1 = strongest]
target_direction                 : +1.0 buy-side target, -1.0 sell-side, 0.0 no target
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base_feature     import BaseFeature
from ..feature_metadata import FeatureMetadata
from ..feature_registry import FeatureRegistry

# ── Default parameters (Pine Script defaults) ─────────────────────────────────
_EQ_TOL     = 0.12   # equal-level tolerance in ATR multiples  (eqTol)
_K_SIDE     = 7      # max pools per side                       (maxPool)
_AGE_MAX    = 400    # bar age threshold for pool removal        (ageMax)
_PROX_K     = 10.0   # ATR multiples at which proximity → 0     (proxK)
_MOM_LEN    = 8      # momentum lookback in bars                 (momLen)
_MOM_K      = 4.0    # ATR multiples at which momentum → 1      (momK)
_W_PROX     = 55.0   # weight: proximity
_W_AGE      = 10.0   # weight: age
_W_TOUCH    = 10.0   # weight: touches
_W_MOM      = 25.0   # weight: momentum
_MIN_SCORE  = 35.0   # minimum score to qualify a pool as NEXT TARGET   (minScore)
_MAX_DIST   = 5.0    # maximum distance (%) for a pool to be NEXT TARGET (maxDist)
_ATR_PERIOD = 14     # ATR smoothing window
_DENSITY_R  = 2.0    # ATR multiples used for liquidity density radius

K = _K_SIDE           # alias used in array slicing throughout


@FeatureRegistry.register
class LiquidityMagnetEngine(BaseFeature):
    """Rank resting liquidity pools by magnetic pull and identify the next target.

    Reads confirmed pivot and EQH/EQL columns from the enriched running_df.
    Never recalculates pivots, swings, BOS/CHoCH, or ATR-based equal levels.
    Maintains its own pool set (7 per side) with momentum-weighted scoring.
    """

    name:             str       = "liquidity_magnet"
    category:         str       = "liquidity"
    dependencies:     list[str] = [
        "market_structure",
        "bos_choch",
        "equal_highs_lows",
        "liquidity_sweeps",          # ordering: run after sweep engine
    ]
    required_columns: list[str] = [
        "open", "high", "low", "close",
        # From market_structure
        "pivot_high", "pivot_low",
        "major_pivot_high", "major_pivot_low",
        # From equal_highs_lows — initial touch-count hint
        "eqh", "eql",
    ]

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912, PLR0914, PLR0915
        """Compute all 20 liquidity magnet features.

        Parameters
        ----------
        df:
            Enriched pipeline DataFrame containing OHLCV plus the columns
            listed in ``required_columns`` (all from upstream generators).

        Returns
        -------
        pd.DataFrame
            20 float64 columns, same index and length as ``df``.
        """
        n   = len(df)
        idx = df.index

        if n == 0:
            return pd.DataFrame(columns=self._output_columns(), index=idx, dtype=float)

        # ── Source arrays ─────────────────────────────────────────────────────
        high_arr = df["high"].to_numpy(dtype=float)
        low_arr  = df["low"].to_numpy(dtype=float)
        cls_arr  = df["close"].to_numpy(dtype=float)

        ph_arr  = df["pivot_high"].to_numpy(dtype=float)
        pl_arr  = df["pivot_low"].to_numpy(dtype=float)
        mph_arr = df["major_pivot_high"].to_numpy(dtype=float)
        mpl_arr = df["major_pivot_low"].to_numpy(dtype=float)
        eqh_arr = df["eqh"].to_numpy(dtype=float)
        eql_arr = df["eql"].to_numpy(dtype=float)

        atr_arr = self._atr(
            df["high"], df["low"], df["close"], _ATR_PERIOD
        ).to_numpy(dtype=float)

        # Momentum lookback: mom_cls[i] = close[i - _MOM_LEN]
        mom_cls = np.empty(n, dtype=float)
        mom_cls[:_MOM_LEN] = cls_arr[0]
        mom_cls[_MOM_LEN:] = cls_arr[:n - _MOM_LEN]

        # ── Pool slot arrays — sell-side (below price) ────────────────────────
        sp_price   = np.zeros(K, dtype=float)
        sp_bar     = np.full(K, -1, dtype=np.int64)
        sp_touches = np.zeros(K, dtype=float)
        sp_major   = np.zeros(K, dtype=bool)
        sp_active  = np.zeros(K, dtype=bool)

        # ── Pool slot arrays — buy-side (above price) ─────────────────────────
        bp_price   = np.zeros(K, dtype=float)
        bp_bar     = np.full(K, -1, dtype=np.int64)
        bp_touches = np.zeros(K, dtype=float)
        bp_major   = np.zeros(K, dtype=bool)
        bp_active  = np.zeros(K, dtype=bool)

        # ── Pre-allocated flat arrays (2K) — filled in-place, no concatenate ──
        TK = 2 * K
        fa_act   = np.empty(TK, dtype=bool)
        fa_price = np.empty(TK, dtype=float)
        fa_score = np.empty(TK, dtype=float)
        fa_touch = np.empty(TK, dtype=float)
        fa_prox  = np.empty(TK, dtype=float)
        fa_age   = np.empty(TK, dtype=float)
        fa_tch   = np.empty(TK, dtype=float)
        fa_mom   = np.empty(TK, dtype=float)
        # fa_is_buy is constant — True for the buy-side (upper) K slots
        fa_is_buy = np.zeros(TK, dtype=bool)
        fa_is_buy[K:] = True

        # Scoring work buffers (re-used each bar, no per-bar allocation)
        _da = np.empty(K, dtype=float)   # distance in ATR
        _pn = np.empty(K, dtype=float)   # prox norm
        _an = np.empty(K, dtype=float)   # age norm
        _tn = np.empty(K, dtype=float)   # touch norm
        _ag = np.empty(K, dtype=float)   # ages (float)

        # ── Output arrays — all initialised to 0.0; NaN for distance cols ─────
        out_cols = self._output_columns()
        out = {c: np.zeros(n, dtype=float) for c in out_cols}
        out["nearest_buy_liquidity_distance"][:] = np.nan
        out["nearest_sell_liquidity_distance"][:] = np.nan

        # ── Bar loop ──────────────────────────────────────────────────────────
        for i in range(n):
            c     = cls_arr[i]
            h_i   = high_arr[i]
            lo_i  = low_arr[i]
            atr_i = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 1e-8
            if atr_i < 1e-8:
                atr_i = 1e-8
            eq_band = _EQ_TOL * atr_i

            # ── A. Age-out stale pools ────────────────────────────────────────
            sp_active[(sp_active) & ((i - sp_bar) > _AGE_MAX)] = False
            bp_active[(bp_active) & ((i - bp_bar) > _AGE_MAX)] = False

            # ── B. Sweep detection (confirmed on close) ───────────────────────
            sp_active[sp_active & (lo_i < sp_price) & (c > sp_price)] = False
            bp_active[bp_active & (h_i > bp_price) & (c < bp_price)] = False

            # ── C. Ingest new pivot high → buy-side pool ──────────────────────
            if ph_arr[i] == 1.0:
                np.abs(bp_price - h_i, out=_da)
                bp_near = bp_active & (_da <= eq_band)
                if bp_near.any():
                    _da[~bp_near] = np.inf
                    bp_touches[int(np.argmin(_da))] += 1.0
                else:
                    self._add_slot(
                        bp_price, bp_bar, bp_touches, bp_major, bp_active,
                        h_i, i, mph_arr[i] == 1.0,
                        2.0 if eqh_arr[i] == 1.0 else 1.0,
                    )

            # ── D. Ingest new pivot low → sell-side pool ──────────────────────
            if pl_arr[i] == 1.0:
                np.abs(sp_price - lo_i, out=_da)
                sp_near = sp_active & (_da <= eq_band)
                if sp_near.any():
                    _da[~sp_near] = np.inf
                    sp_touches[int(np.argmin(_da))] += 1.0
                else:
                    self._add_slot(
                        sp_price, sp_bar, sp_touches, sp_major, sp_active,
                        lo_i, i, mpl_arr[i] == 1.0,
                        2.0 if eql_arr[i] == 1.0 else 1.0,
                    )

            # ── E. Momentum scalars (scalar clip is faster than np.clip) ──────
            raw_mom  = (c - mom_cls[i]) / (atr_i * _MOM_K)
            bull_mom = raw_mom if raw_mom > 0.0 else 0.0
            if bull_mom > 1.0: bull_mom = 1.0
            bear_mom = -raw_mom if raw_mom < 0.0 else 0.0
            if bear_mom > 1.0: bear_mom = 1.0

            # ── F. Fill flat arrays in-place (no np.concatenate) ─────────────
            fa_act[:K]   = sp_active;  fa_act[K:]   = bp_active
            fa_price[:K] = sp_price;   fa_price[K:] = bp_price
            fa_touch[:K] = sp_touches; fa_touch[K:] = bp_touches

            if not fa_act.any():
                continue

            # ── G. Inline score: sell-side (lower K slots) ────────────────────
            np.abs(c - sp_price, out=_da); _da /= atr_i
            np.clip(1.0 - _da / _PROX_K, 0.0, 1.0, out=_pn)
            np.multiply(_pn, _W_PROX, out=fa_prox[:K])

            _ag[:] = i - sp_bar; _ag[~sp_active] = 0.0
            np.clip(1.0 - _ag / _AGE_MAX, 0.0, 1.0, out=_an)
            np.multiply(_an, _W_AGE, out=fa_age[:K])

            np.clip(sp_touches / 5.0, 0.0, 1.0, out=_tn)
            np.multiply(_tn, _W_TOUCH, out=fa_tch[:K])

            fa_mom[:K] = bear_mom * _W_MOM
            np.add(fa_prox[:K], fa_age[:K], out=fa_score[:K])
            fa_score[:K] += fa_tch[:K]; fa_score[:K] += fa_mom[:K]
            fa_score[:K] = np.where(sp_active, fa_score[:K], -1.0)

            # ── H. Inline score: buy-side (upper K slots) ────────────────────
            np.abs(c - bp_price, out=_da); _da /= atr_i
            np.clip(1.0 - _da / _PROX_K, 0.0, 1.0, out=_pn)
            np.multiply(_pn, _W_PROX, out=fa_prox[K:])

            _ag[:] = i - bp_bar; _ag[~bp_active] = 0.0
            np.clip(1.0 - _ag / _AGE_MAX, 0.0, 1.0, out=_an)
            np.multiply(_an, _W_AGE, out=fa_age[K:])

            np.clip(bp_touches / 5.0, 0.0, 1.0, out=_tn)
            np.multiply(_tn, _W_TOUCH, out=fa_tch[K:])

            fa_mom[K:] = bull_mom * _W_MOM
            np.add(fa_prox[K:], fa_age[K:], out=fa_score[K:])
            fa_score[K:] += fa_tch[K:]; fa_score[K:] += fa_mom[K:]
            fa_score[K:] = np.where(bp_active, fa_score[K:], -1.0)

            # ── I. Distance from close to each pool (in %) ───────────────────
            dist_pct = np.where(fa_act, np.abs(c - fa_price) / (c + 1e-12) * 100.0, np.inf)

            # ── J. Nearest pool overall ───────────────────────────────────────
            nearest_k  = int(np.argmin(dist_pct))
            near_sc    = float(fa_score[nearest_k]) if fa_score[nearest_k] >= 0 else 0.0
            out["nearest_liquidity_score"][i] = near_sc
            out["nearest_cluster_size"][i]    = float(fa_touch[nearest_k])

            # Nearest per-side
            buy_dists  = np.where(fa_act & fa_is_buy,  dist_pct, np.inf)
            sell_dists = np.where(fa_act & ~fa_is_buy, dist_pct, np.inf)
            bd_finite  = buy_dists[np.isfinite(buy_dists)]
            sd_finite  = sell_dists[np.isfinite(sell_dists)]
            if bd_finite.size:
                out["nearest_buy_liquidity_distance"][i]  = float(bd_finite.min())
            if sd_finite.size:
                out["nearest_sell_liquidity_distance"][i] = float(sd_finite.min())

            # ── K. Magnet strength ────────────────────────────────────────────
            active_sc = fa_score[fa_act]
            out["magnet_strength"][i] = float(active_sc.max())

            # ── L. Side-weighted probabilities ───────────────────────────────
            buy_sum  = float(np.sum(np.where(fa_act & fa_is_buy,
                                             np.clip(fa_score, 0, None), 0.0)))
            sell_sum = float(np.sum(np.where(fa_act & ~fa_is_buy,
                                             np.clip(fa_score, 0, None), 0.0)))
            tot = buy_sum + sell_sum
            if tot > 0.0:
                out["buy_side_probability"][i]  = buy_sum  / tot
                out["sell_side_probability"][i] = sell_sum / tot

            # ── M. Liquidity density ──────────────────────────────────────────
            density_pct = _DENSITY_R * atr_i / (c + 1e-12) * 100.0
            out["liquidity_density"][i] = float(np.sum(fa_act & (dist_pct <= density_pct)))

            # ── N. Ranking position of nearest pool ───────────────────────────
            out["ranking_position"][i] = float(int(np.sum(active_sc > fa_score[nearest_k])) + 1)

            # ── O. Target selection ───────────────────────────────────────────
            target_mask = fa_act & (dist_pct <= _MAX_DIST) & (fa_score >= _MIN_SCORE)
            if target_mask.any():
                t_k   = int(np.argmax(np.where(target_mask, fa_score, -1.0)))
                t_sc  = float(fa_score[t_k])
                t_pr  = float(fa_price[t_k])
                t_d   = float(dist_pct[t_k])
                t_buy = bool(fa_is_buy[t_k])

                out["magnet_score"][i]           = t_sc
                out["magnet_probability"][i]     = t_sc / 100.0
                out["target_liquidity"][i]       = t_pr
                out["distance_to_target"][i]     = t_d
                out["cluster_strength"][i]       = float(fa_touch[t_k])
                out["target_direction"][i]       = 1.0 if t_buy else -1.0
                out["proximity_contribution"][i] = float(fa_prox[t_k])
                out["age_contribution"][i]       = float(fa_age[t_k])
                out["touch_contribution"][i]     = float(fa_tch[t_k])
                out["momentum_contribution"][i]  = float(fa_mom[t_k])
                out["liquidity_rank"][i]         = float(int(np.sum(active_sc > t_sc)) + 1)

        return pd.DataFrame(out, index=idx)

    # ── Scoring utility (used by tests; hot-path is inlined in generate()) ────

    @staticmethod
    def _score(
        pool:        dict,
        current_bar: int,
        close:       float,
        atr_i:       float,
        dir_mom:     float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reference scoring implementation for a single-side pool dict.

        Accepts a dict with keys ``price``, ``bar``, ``touches``, ``active``
        (each a numpy array of length K).  Returns (prox_c, age_c, touch_c,
        total_score) where inactive slots get total = -1.0.

        Not called from ``generate()`` (scoring is inlined for performance).
        Exposed as a static method so unit tests can verify component values.
        """
        act     = pool["active"]
        prices  = pool["price"]
        bars    = pool["bar"]
        touches = pool["touches"]

        dist_atr  = np.abs(close - prices) / atr_i
        prox_c    = np.clip(1.0 - dist_atr / _PROX_K, 0.0, 1.0) * _W_PROX

        ages    = np.where(act, current_bar - bars, 0).astype(float)
        age_c   = np.clip(1.0 - ages / _AGE_MAX, 0.0, 1.0) * _W_AGE

        touch_c = np.clip(touches / 5.0, 0.0, 1.0) * _W_TOUCH

        k_len   = len(prices)
        mom_c   = np.full(k_len, dir_mom * _W_MOM, dtype=float)

        total   = np.where(act, prox_c + age_c + touch_c + mom_c, -1.0)
        return prox_c, age_c, touch_c, total

    # ── Slot management ───────────────────────────────────────────────────────

    @staticmethod
    def _add_slot(
        prices: np.ndarray, bars: np.ndarray, touches: np.ndarray,
        major: np.ndarray, active: np.ndarray,
        price: float, bar: int, is_major: bool, init_touches: float,
    ) -> None:
        """Insert a pool into the first empty slot; evict the oldest if full."""
        empty = np.where(~active)[0]
        k = int(empty[0]) if len(empty) > 0 else int(np.argmin(bars))
        prices[k]  = price
        bars[k]    = bar
        touches[k] = init_touches
        major[k]   = is_major
        active[k]  = True

    # ── ATR helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int,
    ) -> pd.Series:
        prev = close.shift(1)
        tr   = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    # ── Metadata helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _output_columns() -> list[str]:
        return [
            "nearest_buy_liquidity_distance",
            "nearest_sell_liquidity_distance",
            "nearest_liquidity_score",
            "magnet_score",
            "magnet_probability",
            "liquidity_rank",
            "target_liquidity",
            "distance_to_target",
            "buy_side_probability",
            "sell_side_probability",
            "liquidity_density",
            "cluster_strength",
            "magnet_strength",
            "nearest_cluster_size",
            "proximity_contribution",
            "age_contribution",
            "touch_contribution",
            "momentum_contribution",
            "ranking_position",
            "target_direction",
        ]

    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name        = self.name,
            category    = self.category,
            description = (
                "LunqFX Liquidity Magnet — ranks every resting liquidity pool 0-100 "
                "by how strongly it is likely to be swept next.  "
                "Score = proximity (55 pts) + momentum (25 pts) + age (10 pts) + "
                "touches (10 pts).  The highest-scoring pool within 5 % of close and "
                "above 35 pts is named the NEXT TARGET with a probability score.  "
                "All outputs are numerical ML features."
            ),
            dependencies     = self.dependencies,
            required_columns = self.required_columns,
            output_columns   = self._output_columns(),
            version    = "1.0.0",
            author     = "Smart Trading Team",
            complexity = "high",
            tags       = [
                "ICT", "smart_money", "liquidity",
                "magnet", "ranking", "probability", "sweep_target",
                "buy_side", "sell_side", "momentum", "LunqFX",
            ],
        )
