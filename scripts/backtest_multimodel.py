"""
Multi-Model Backtest — EURUSD M15
Compares three strategies on truly out-of-sample data:

  A. 1b model alone      — current baseline (15-min direction)
  B. HIGH_CONVICTION     — all three models agree on same direction
  C. STRUCTURAL 4b+8b    — 4b + 8b agree (includes SETUP_FORMING entries)

Starting capital: $10,000  |  Fixed risk: $100/trade  |  RR: 2:1
Out-of-sample period: 2024-09-21 → present  (never seen by any model)

Run:
  python scripts/backtest_multimodel.py
"""
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
OOS_START     = pd.Timestamp("2024-09-21", tz="UTC")
OOS_END       = pd.Timestamp("2026-01-01", tz="UTC")
STARTING_CAP  = 10_000.0
RISK_DOLLARS  = 100.0          # fixed $100 risk per trade (1% of starting cap)
TP_ATR_MULT   = 3.0
SL_ATR_MULT   = 1.5
RR            = TP_ATR_MULT / SL_ATR_MULT   # 2.0
MIN_CONF_1B   = 0.60           # min session-adjusted confidence for 1b signal
MAX_HOLD_BARS = 24             # max 6 hours open, then close at current price
PIP           = 0.0001

CLASS = {0: "SELL", 1: "HOLD", 2: "BUY"}


# ── Session weighting ─────────────────────────────────────────────────────────

def session_mult(utc_hour: int) -> float:
    if 7  <= utc_hour <  9:  return 1.00   # LONDON_OPEN
    if 13 <= utc_hour < 17:  return 1.00   # OVERLAP
    if 9  <= utc_hour < 13:  return 0.92   # LONDON
    if 17 <= utc_hour < 20:  return 0.88   # NEW_YORK
    if 15 <= utc_hour < 17:  return 0.85   # LONDON_CLOSE
    if  0 <= utc_hour <  7:  return 0.80   # ASIAN
    if 20 <= utc_hour < 22:  return 0.72   # NY_CLOSE
    return 0.60                             # DEAD_ZONE


def adj_confidence(p_sell, p_hold, p_buy, direction, mult):
    """Return session-adjusted directional confidence."""
    if direction == "SELL":
        return p_sell * mult
    elif direction == "BUY":
        return p_buy * mult
    return p_hold   # HOLD direction


def is_weekend(ts: pd.Timestamp) -> bool:
    dow = ts.dayofweek
    return dow == 6 or (dow == 5 and ts.hour >= 22)


# ── Model loader ──────────────────────────────────────────────────────────────

class ModelBundle:
    def __init__(self, bundle_dir: Path):
        self.model   = joblib.load(bundle_dir / "model.joblib")
        self.preproc = joblib.load(bundle_dir / "preprocessing.joblib")
        self.features = json.loads((bundle_dir / "feature_order.json").read_text())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = X.reindex(columns=self.features)   # fill missing with NaN
        Xp = self.preproc.transform(Xc)
        return self.model.predict_proba(Xp)      # (n, 3): [sell, hold, buy]


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_bar:   int
    exit_bar:    int   = -1
    direction:   str   = ""
    entry_price: float = 0.0
    tp_price:    float = 0.0
    sl_price:    float = 0.0
    atr_pips:    float = 0.0
    outcome:     str   = ""    # "WIN" | "LOSS" | "TIMEOUT"
    pnl:         float = 0.0
    entry_ts:    Optional[pd.Timestamp] = None
    conviction:  str   = ""    # "HIGH_CONVICTION" | "SETUP_FORMING" | "DIRECTIONAL_BIAS"


# ── Core simulation (one-pass, correct in-trade tracking) ─────────────────────

def run_strategy(
    name:             str,
    df:               pd.DataFrame,
    proba_1b:         np.ndarray,
    proba_4b:         np.ndarray,
    proba_8b:         np.ndarray,
    use_1b_only:      bool = False,
    require_all:      bool = False,
    structural_only:  bool = False,
) -> tuple[list[Trade], float]:
    """
    Single-pass simulation with correct in-trade tracking.
    Returns (list_of_trades, final_equity).
    """
    trades: list[Trade] = []
    equity       = STARTING_CAP
    open_trade: Optional[Trade] = None

    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atrs   = df["atr"].to_numpy(dtype=float)
    n      = len(df)

    for i in range(n):
        row = df.iloc[i]
        ts  = row["timestamp"]

        # ── 1. Manage open trade exit ────────────────────────────────
        if open_trade is not None:
            hi, lo, cl = highs[i], lows[i], closes[i]
            bars_held = i - open_trade.entry_bar
            outcome = None

            if open_trade.direction == "BUY":
                if hi >= open_trade.tp_price:   outcome = "WIN"
                elif lo <= open_trade.sl_price: outcome = "LOSS"
            else:
                if lo <= open_trade.tp_price:   outcome = "WIN"
                elif hi >= open_trade.sl_price: outcome = "LOSS"

            if outcome is None and bars_held >= MAX_HOLD_BARS:
                outcome = "TIMEOUT"

            if outcome is not None:
                open_trade.outcome  = outcome
                open_trade.exit_bar = i
                if outcome == "WIN":
                    open_trade.pnl = +RISK_DOLLARS * RR
                elif outcome == "LOSS":
                    open_trade.pnl = -RISK_DOLLARS
                else:  # TIMEOUT: actual pip move, sized the same way
                    move_pips = (cl - open_trade.entry_price) / PIP
                    if open_trade.direction == "SELL":
                        move_pips = -move_pips
                    open_trade.pnl = RISK_DOLLARS * (move_pips / open_trade.atr_pips / SL_ATR_MULT)
                equity += open_trade.pnl
                trades.append(open_trade)
                open_trade = None

        # ── 2. Skip weekends or if still in trade ───────────────────
        if open_trade is not None or is_weekend(ts):
            continue

        # ── 3. Signal generation ─────────────────────────────────────
        p1 = proba_1b[i]; p4 = proba_4b[i]; p8 = proba_8b[i]
        dir_1b = CLASS[int(np.argmax(p1))]
        dir_4b = CLASS[int(np.argmax(p4))]
        dir_8b = CLASS[int(np.argmax(p8))]

        mult   = session_mult(ts.hour)
        conf_1 = adj_confidence(p1[0], p1[1], p1[2], dir_1b, mult)

        trade_dir: Optional[str] = None
        conviction_label = ""

        if use_1b_only:
            if dir_1b != "HOLD" and conf_1 >= MIN_CONF_1B:
                trade_dir = dir_1b
                conviction_label = "1B_SIGNAL"

        elif require_all:
            if (dir_1b != "HOLD" and dir_4b != "HOLD" and dir_8b != "HOLD"
                    and dir_1b == dir_4b == dir_8b
                    and conf_1 >= MIN_CONF_1B):
                trade_dir = dir_1b
                conviction_label = "HIGH_CONVICTION"

        elif structural_only:
            if dir_4b != "HOLD" and dir_8b != "HOLD" and dir_4b == dir_8b:
                trade_dir = dir_4b
                conviction_label = "HIGH_CONVICTION" if dir_1b == dir_4b else "SETUP_FORMING"

        if trade_dir is None:
            continue

        # ── 4. Open trade ─────────────────────────────────────────────
        entry = closes[i]
        atr   = atrs[i]
        atr_p = atr / PIP if atr > 0 else 10.0   # ATR in pips

        if trade_dir == "BUY":
            tp = entry + atr * TP_ATR_MULT
            sl = entry - atr * SL_ATR_MULT
        else:
            tp = entry - atr * TP_ATR_MULT
            sl = entry + atr * SL_ATR_MULT

        open_trade = Trade(
            entry_bar=i, direction=trade_dir, entry_price=entry,
            tp_price=tp, sl_price=sl, atr_pips=atr_p,
            entry_ts=ts, conviction=conviction_label,
        )

    # Force-close any trade still open at end of data
    if open_trade is not None:
        cl = closes[-1]
        move_pips = (cl - open_trade.entry_price) / PIP
        if open_trade.direction == "SELL":
            move_pips = -move_pips
        open_trade.outcome  = "TIMEOUT"
        open_trade.exit_bar = n - 1
        open_trade.pnl = RISK_DOLLARS * (move_pips / open_trade.atr_pips / SL_ATR_MULT)
        equity += open_trade.pnl
        trades.append(open_trade)

    return trades, equity


# ── Report helper ─────────────────────────────────────────────────────────────

def report(name: str, trades: list[Trade], final_equity: float) -> dict:
    if not trades:
        print(f"\n  {name}: NO TRADES")
        return {}

    wins    = [t for t in trades if t.outcome == "WIN"]
    losses  = [t for t in trades if t.outcome == "LOSS"]
    timeout = [t for t in trades if t.outcome == "TIMEOUT"]

    win_rate = len(wins) / len(trades) * 100
    total_pnl = sum(t.pnl for t in trades)
    gross_w   = sum(t.pnl for t in wins)
    gross_l   = abs(sum(t.pnl for t in losses))
    pf        = gross_w / gross_l if gross_l > 0 else float("inf")

    # Max drawdown (running equity)
    equity_curve = [STARTING_CAP]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t.pnl)
    peak = STARTING_CAP
    mdd  = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        mdd  = max(mdd, (peak - eq) / peak * 100)

    avg_bars = np.mean([t.exit_bar - t.entry_bar for t in trades])

    print(f"\n  {'─'*57}")
    print(f"  Strategy : {name}")
    print(f"  {'─'*57}")
    print(f"  Trades   : {len(trades):>4}  (W={len(wins)} L={len(losses)} T={len(timeout)})")
    print(f"  Win Rate : {win_rate:.1f}%   (break-even: {1/(1+RR)*100:.0f}%)")
    print(f"  Prof Fac : {pf:.2f}")
    print(f"  Total P&L: ${total_pnl:>+8,.0f}   Final: ${final_equity:>8,.0f}")
    print(f"  Return   : {(final_equity/STARTING_CAP - 1)*100:+.1f}%")
    print(f"  Max DD   : {mdd:.1f}%")
    print(f"  Avg Hold : {avg_bars:.1f} bars  ({avg_bars*15/60:.1f} hrs)")
    print(f"  Win P&L  : ${RISK_DOLLARS*RR:+.0f} each    Loss P&L: ${-RISK_DOLLARS:.0f} each  (fixed risk)")

    return {
        "name": name, "trades": len(trades), "wins": len(wins),
        "losses": len(losses), "win_rate": win_rate,
        "profit_factor": pf, "total_pnl": total_pnl,
        "final_equity": final_equity, "max_dd": mdd,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MFIP — Multi-Model Backtest  (Out-of-Sample)")
    print(f"  Period: {OOS_START.date()} → {OOS_END.date()}")
    print(f"  Capital: ${STARTING_CAP:,.0f}   Risk: ${RISK_DOLLARS}/trade (fixed)")
    print(f"  TP={TP_ATR_MULT}×ATR   SL={SL_ATR_MULT}×ATR   RR={RR:.1f}:1")
    print(f"  Win=$+{RISK_DOLLARS*RR:.0f}   Loss=$-{RISK_DOLLARS:.0f}   Break-even={1/(1+RR)*100:.0f}% WR")
    print("=" * 65)

    # ── Load data ────────────────────────────────────────────────────
    print("\nLoading feature dataset...")
    df = pd.read_parquet(ROOT / "data/features/EURUSD/feature_dataset.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)
    oos  = df[mask].reset_index(drop=True)
    print(f"  OOS period: {len(oos):,} bars")
    print(f"  {oos['timestamp'].iloc[0].date()} → {oos['timestamp'].iloc[-1].date()}")

    # ── Load models ──────────────────────────────────────────────────
    print("\nLoading models...")
    m1 = ModelBundle(ROOT / "models/best_model")
    m4 = ModelBundle(ROOT / "models/lookahead_4b")
    m8 = ModelBundle(ROOT / "models/lookahead_8b")
    print("  best_model (1b), lookahead_4b, lookahead_8b  — all loaded")

    # ── Batch inference ──────────────────────────────────────────────
    print("\nRunning batch inference on OOS data...")
    p1 = m1.predict_proba(oos)
    p4 = m4.predict_proba(oos)
    p8 = m8.predict_proba(oos)

    dirs1 = np.array([CLASS[int(np.argmax(r))] for r in p1])
    dirs4 = np.array([CLASS[int(np.argmax(r))] for r in p4])
    dirs8 = np.array([CLASS[int(np.argmax(r))] for r in p8])

    print(f"  1b signals: BUY={( dirs1=='BUY').sum()} SELL={(dirs1=='SELL').sum()} HOLD={(dirs1=='HOLD').sum()}")
    print(f"  4b signals: BUY={(dirs4=='BUY').sum()} SELL={(dirs4=='SELL').sum()} HOLD={(dirs4=='HOLD').sum()}")
    print(f"  8b signals: BUY={(dirs8=='BUY').sum()} SELL={(dirs8=='SELL').sum()} HOLD={(dirs8=='HOLD').sum()}")

    # Agreement stats
    all_agree     = ((dirs1==dirs4) & (dirs4==dirs8) & (dirs1!="HOLD")).sum()
    struct_agree  = ((dirs4==dirs8) & (dirs4!="HOLD")).sum()
    print(f"  All-3 agree (directional): {all_agree} bars")
    print(f"  4b+8b agree (structural) : {struct_agree} bars")

    # ── Run strategies ───────────────────────────────────────────────
    print("\nSimulating trades (one pass, correct in-trade tracking)...")
    trades_A, eq_A = run_strategy("A: 1b Alone",        oos, p1, p4, p8, use_1b_only=True)
    trades_B, eq_B = run_strategy("B: High Conviction",  oos, p1, p4, p8, require_all=True)
    trades_C, eq_C = run_strategy("C: Structural 4b+8b", oos, p1, p4, p8, structural_only=True)

    # ── Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    r_A = report("A: 1b Alone",        trades_A, eq_A)
    r_B = report("B: High Conviction",  trades_B, eq_B)
    r_C = report("C: Structural 4b+8b", trades_C, eq_C)

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMPARISON TABLE")
    print("=" * 65)
    print(f"  {'Strategy':<24} {'Trades':>7} {'WR%':>6} {'PF':>6} {'P&L':>9} {'Final':>9} {'DD%':>6}")
    print(f"  {'─'*24} {'─'*7} {'─'*6} {'─'*6} {'─'*9} {'─'*9} {'─'*6}")
    for r in [r_A, r_B, r_C]:
        if r:
            print(
                f"  {r['name']:<24} {r['trades']:>7} {r['win_rate']:>5.1f}%"
                f" {r['profit_factor']:>6.2f} ${r['total_pnl']:>+8,.0f}"
                f" ${r['final_equity']:>8,.0f} {r['max_dd']:>5.1f}%"
            )

    # ── Strategy C breakdown ─────────────────────────────────────────
    hc = [t for t in trades_C if t.conviction == "HIGH_CONVICTION"]
    sf = [t for t in trades_C if t.conviction == "SETUP_FORMING"]

    def wr(lst):
        if not lst: return 0.0
        return sum(1 for t in lst if t.outcome == "WIN") / len(lst) * 100

    def pnl(lst):
        return sum(t.pnl for t in lst)

    print(f"\n  Strategy C sub-breakdown:")
    print(f"  {'Level':<20} {'Count':>6} {'WR%':>7} {'P&L':>10}")
    print(f"  {'─'*20} {'─'*6} {'─'*7} {'─'*10}")
    print(f"  {'HIGH_CONVICTION':<20} {len(hc):>6} {wr(hc):>6.1f}% ${pnl(hc):>+9,.0f}")
    print(f"  {'SETUP_FORMING':<20} {len(sf):>6} {wr(sf):>6.1f}% ${pnl(sf):>+9,.0f}")

    # ── Month-by-month P&L for Strategy A vs B ────────────────────────
    print(f"\n  Monthly P&L summary (A vs B):")
    print(f"  {'Month':<10} {'A: Trades':>10} {'A: P&L':>9} {'B: Trades':>10} {'B: P&L':>9} {'B Win%':>7}")
    print(f"  {'─'*10} {'─'*10} {'─'*9} {'─'*10} {'─'*9} {'─'*7}")

    months_A = {}
    for t in trades_A:
        k = t.entry_ts.strftime("%Y-%m")
        months_A.setdefault(k, []).append(t)
    months_B = {}
    for t in trades_B:
        k = t.entry_ts.strftime("%Y-%m")
        months_B.setdefault(k, []).append(t)

    all_months = sorted(set(list(months_A.keys()) + list(months_B.keys())))
    for m in all_months:
        ta = months_A.get(m, [])
        tb = months_B.get(m, [])
        pa = sum(x.pnl for x in ta)
        pb = sum(x.pnl for x in tb)
        wb = wr(tb)
        print(f"  {m:<10} {len(ta):>10} ${pa:>+8,.0f} {len(tb):>10} ${pb:>+8,.0f} {wb:>6.1f}%")

    print()


if __name__ == "__main__":
    main()
