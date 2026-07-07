"""Market Regime Analyzer — Consolidation / Expansion / Manipulation.

Uses the 247-feature dataset already produced by build_inference_features()
to classify the current market state using ICT/SMC concepts and technical
volatility indicators.

Typical usage
-------------
    from src.inference.feature_builder   import build_inference_features
    from src.inference.market_regime      import analyze_market_regime, print_regime_report

    feat_df = build_inference_features(m15_df, htf_dfs={...})
    report  = analyze_market_regime(feat_df)
    print_regime_report(report)

Three regimes (can overlap):
    CONSOLIDATION  — price coiling, no clear direction, accumulation/distribution
    EXPANSION      — impulse move, trend continuation, break of structure
    MANIPULATION   — liquidity sweep, stop hunt, false breakout before reversal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
ADX_TREND_THRESHOLD    = 25.0   # ADX below this = no trend (consolidation)
ADX_STRONG_THRESHOLD   = 40.0   # ADX above this = strong trend (expansion)
BB_NARROW_PERCENTILE   = 25     # BB width in bottom 25% of history = squeeze
BB_WIDE_PERCENTILE     = 75     # BB width in top 75% of history = expansion
ATR_COMPRESS_MULT      = 0.70   # ATR < 70% of 20-bar avg = compression
ATR_EXPAND_MULT        = 1.30   # ATR > 130% of 20-bar avg = expansion
NOISE_HIGH_THRESHOLD   = 0.65   # market_noise > 0.65 = choppy (consolidation)
EFFICIENCY_LOW         = 0.30   # efficiency_ratio < 0.30 = consolidation


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class RegimeSignal:
    """One piece of evidence for a regime classification."""
    name:        str
    value:       object
    description: str
    weight:      float   # 0.0 – 1.0, how strongly this signal votes for the regime


@dataclass
class RegimeReport:
    """Full market regime assessment for the current bar."""

    # ── Primary classification ───────────────────────────────────────────────
    dominant_regime:   str              # "CONSOLIDATION" | "EXPANSION" | "MANIPULATION"
    regime_scores:     dict[str, float] # {regime: score 0-1}

    # ── Market context ───────────────────────────────────────────────────────
    timestamp:         object
    close:             Optional[float]
    atr_pips:          Optional[float]
    atr_vs_avg:        Optional[str]    # "COMPRESSED" | "NORMAL" | "EXPANDED"
    bias:              str              # "BULLISH" | "BEARISH" | "NEUTRAL"
    pd_zone:           str              # "PREMIUM" | "DISCOUNT" | "EQUILIBRIUM"
    trend_strength:    Optional[float]
    adx:               Optional[float]

    # ── ICT/SMC specific signals ──────────────────────────────────────────────
    liquidity_sweep:   bool
    sweep_direction:   str              # "BULLISH" | "BEARISH" | "NONE"
    sweep_rejected:    bool
    sweep_confirmed:   bool
    choch_detected:    bool
    choch_direction:   str              # "BULLISH" | "BEARISH" | "NONE"
    bos_detected:      bool
    bos_direction:     str              # "BULLISH" | "BEARISH" | "NONE"
    fvg_active:        bool
    fvg_direction:     str              # "BULLISH" | "BEARISH" | "NONE"
    ob_active:         bool
    ob_direction:      str              # "BULLISH" | "BEARISH" | "NONE"
    in_order_block:    bool
    # OB price levels — exact zone of the most recent active OB (None if no OB)
    ob_bullish_top:    Optional[float]
    ob_bullish_bottom: Optional[float]
    ob_bearish_top:    Optional[float]
    ob_bearish_bottom: Optional[float]

    # ── Supporting evidence ───────────────────────────────────────────────────
    consolidation_signals: list[RegimeSignal] = field(default_factory=list)
    expansion_signals:     list[RegimeSignal] = field(default_factory=list)
    manipulation_signals:  list[RegimeSignal] = field(default_factory=list)

    # ── Interpretation ───────────────────────────────────────────────────────
    narrative:         str = ""         # Human-readable summary
    trade_implication: str = ""         # What this regime means for entries


# ── Main function ─────────────────────────────────────────────────────────────

def analyze_market_regime(
    feature_df: pd.DataFrame,
    lookback:   int = 1,
) -> RegimeReport:
    """Classify the market regime for the most recent bar(s).

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of ``build_inference_features()``.
    lookback : int
        Number of most recent bars to analyse.  Default 1 (latest bar only).
        When > 1, the regime is assessed on the last *lookback* rows and the
        dominant regime across that window is returned.

    Returns
    -------
    RegimeReport
    """
    if feature_df.empty:
        raise ValueError("feature_df is empty — cannot assess regime")

    # Use the last `lookback` rows (most recent bars)
    df = feature_df.tail(lookback).copy()
    row = df.iloc[-1]   # latest bar for single-value reads

    # ── Helper: safe scalar read ──────────────────────────────────────────────
    def get(col: str, default=None):
        if col in row.index:
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return v
        return default

    def get_bool(col: str) -> bool:
        v = get(col, 0)
        return bool(v) if v is not None else False

    def get_float(col: str, default: float = 0.0) -> float:
        v = get(col, default)
        try:
            return float(v)
        except Exception:
            return default

    # ── Extract raw values ────────────────────────────────────────────────────
    timestamp  = get("timestamp")
    close      = get_float("close")
    atr        = get_float("atr", 0.0008)
    atr_pips   = round(atr / 0.0001, 1) if atr else None

    # ATR vs rolling average (detect compression / expansion)
    atr_20     = get_float("rolling_atr_20", atr)
    if atr_20 > 0:
        atr_ratio = atr / atr_20
        atr_vs_avg = ("COMPRESSED" if atr_ratio < ATR_COMPRESS_MULT
                      else "EXPANDED" if atr_ratio > ATR_EXPAND_MULT
                      else "NORMAL")
    else:
        atr_ratio, atr_vs_avg = 1.0, "NORMAL"

    adx            = get_float("adx",              0.0)
    bb_width       = get_float("bb_width",          0.0)
    market_noise   = get_float("market_noise",      0.5)
    eff_ratio      = get_float("efficiency_ratio",  0.5)
    trend_score    = get_float("trend_score",       0.0)
    trend_strength = get_float("trend_strength",    0.0)
    vol_regime     = get("volatility_regime", "normal")
    vol_compress   = get_bool("volatility_compression")
    vol_expand     = get_bool("volatility_expansion")
    structure_bias = get("structure_bias", "neutral")
    pd_zone_raw    = get("pd_zone", 0)

    # ICT/SMC signals
    bull_sweep     = get_bool("bullish_liquidity_sweep")
    bear_sweep     = get_bool("bearish_liquidity_sweep")
    sweep_reject   = get_bool("sweep_rejection")
    sweep_confirm  = get_bool("confirmed_sweep")
    sweep_strength = get_float("sweep_strength", 0.0)
    choch_bull     = get_bool("choch_bullish")
    choch_bear     = get_bool("choch_bearish")
    bos_bull       = get_bool("bos_bullish")
    bos_bear       = get_bool("bos_bearish")
    fvg_bull_act   = get_bool("fvg_bullish_active")
    fvg_bear_act   = get_bool("fvg_bearish_active")
    ob_bull_act    = get_bool("ob_bullish_active")
    ob_bear_act    = get_bool("ob_bearish_active")
    in_bull_ob     = get_bool("price_in_bullish_ob")
    in_bear_ob     = get_bool("price_in_bearish_ob")

    # OB price bands (NaN → None)
    def get_price(col: str) -> Optional[float]:
        v = get(col)
        if v is None:
            return None
        try:
            f = float(v)
            return None if (f != f) else round(f, 5)  # nan check
        except Exception:
            return None

    ob_bullish_top    = get_price("ob_bullish_top")
    ob_bullish_bottom = get_price("ob_bullish_bottom")
    ob_bearish_top    = get_price("ob_bearish_top")
    ob_bearish_bottom = get_price("ob_bearish_bottom")

    # Derived booleans
    liquidity_sweep = bull_sweep or bear_sweep
    sweep_direction = ("BULLISH" if bull_sweep else "BEARISH" if bear_sweep else "NONE")
    choch_detected  = choch_bull or choch_bear
    choch_direction = ("BULLISH" if choch_bull else "BEARISH" if choch_bear else "NONE")
    bos_detected    = bos_bull or bos_bear
    bos_direction   = ("BULLISH" if bos_bull else "BEARISH" if bos_bear else "NONE")
    fvg_active      = fvg_bull_act or fvg_bear_act
    fvg_direction   = ("BULLISH" if fvg_bull_act else "BEARISH" if fvg_bear_act else "NONE")
    ob_active       = ob_bull_act or ob_bear_act
    ob_direction    = ("BULLISH" if ob_bull_act else "BEARISH" if ob_bear_act else "NONE")
    in_order_block  = in_bull_ob or in_bear_ob

    # Bias & Premium/Discount zone
    bias_map = {"bullish": "BULLISH", "bearish": "BEARISH"}
    bias = bias_map.get(str(structure_bias).lower(), "NEUTRAL")
    try:
        pd_v = float(pd_zone_raw)
        pd_zone = ("PREMIUM" if pd_v > 0.1 else "DISCOUNT" if pd_v < -0.1 else "EQUILIBRIUM")
    except Exception:
        pd_zone = "EQUILIBRIUM"

    # ── Build evidence lists ───────────────────────────────────────────────────
    con_sigs: list[RegimeSignal] = []
    exp_sigs: list[RegimeSignal] = []
    man_sigs: list[RegimeSignal] = []

    # --- CONSOLIDATION signals ---
    if adx > 0 and adx < ADX_TREND_THRESHOLD:
        con_sigs.append(RegimeSignal(
            "ADX", round(adx, 1),
            f"ADX={adx:.1f} < {ADX_TREND_THRESHOLD} — no directional trend",
            weight=0.8,
        ))
    if vol_compress or atr_vs_avg == "COMPRESSED":
        con_sigs.append(RegimeSignal(
            "ATR compression", f"{atr_pips:.1f} pips ({atr_ratio:.0%} of avg)",
            "ATR below 20-bar average — volatility contracting",
            weight=0.7,
        ))
    if bb_width > 0:
        # Compute rolling percentile of bb_width if enough rows
        if len(df) >= 20 and "bb_width" in df.columns:
            pct = (df["bb_width"] <= bb_width).mean()
        else:
            pct = 0.5
        if pct < (BB_NARROW_PERCENTILE / 100):
            con_sigs.append(RegimeSignal(
                "BB squeeze", f"width={bb_width:.5f} (bot {pct*100:.0f}%ile)",
                "Bollinger Bands narrowing — squeeze setup",
                weight=0.75,
            ))
    if market_noise > NOISE_HIGH_THRESHOLD:
        con_sigs.append(RegimeSignal(
            "Market noise", round(market_noise, 2),
            f"Noise={market_noise:.2f} > {NOISE_HIGH_THRESHOLD} — choppy, no follow-through",
            weight=0.6,
        ))
    if eff_ratio < EFFICIENCY_LOW:
        con_sigs.append(RegimeSignal(
            "Efficiency ratio", round(eff_ratio, 2),
            f"ER={eff_ratio:.2f} — price oscillating, low net displacement",
            weight=0.65,
        ))
    if pd_zone == "EQUILIBRIUM":
        con_sigs.append(RegimeSignal(
            "Premium/Discount zone", "EQUILIBRIUM",
            "Price at 50% of swing range — fair value, no bias",
            weight=0.4,
        ))

    # --- EXPANSION signals ---
    if adx >= ADX_TREND_THRESHOLD:
        con_sigs_weight = 0.0
        exp_sigs.append(RegimeSignal(
            "ADX", round(adx, 1),
            f"ADX={adx:.1f} >= {ADX_TREND_THRESHOLD} — trending market",
            weight=min(1.0, (adx - ADX_TREND_THRESHOLD) / (ADX_STRONG_THRESHOLD - ADX_TREND_THRESHOLD) * 0.8 + 0.2),
        ))
    if vol_expand or atr_vs_avg == "EXPANDED":
        exp_sigs.append(RegimeSignal(
            "ATR expansion", f"{atr_pips:.1f} pips ({atr_ratio:.0%} of avg)",
            "ATR above 20-bar average — volatility expanding",
            weight=0.75,
        ))
    if bos_detected:
        exp_sigs.append(RegimeSignal(
            "Break of Structure", bos_direction,
            f"BOS {bos_direction} confirmed — structural shift, expect continuation",
            weight=0.85,
        ))
    if trend_score > 0.6:
        exp_sigs.append(RegimeSignal(
            "Trend score", round(trend_score, 2),
            f"trend_score={trend_score:.2f} — strong directional momentum",
            weight=0.7,
        ))
    if trend_strength > 0.5:
        exp_sigs.append(RegimeSignal(
            "Trend strength", round(trend_strength, 2),
            "Price closing consistently in direction of trend",
            weight=0.6,
        ))
    if fvg_active:
        exp_sigs.append(RegimeSignal(
            "Fair Value Gap", fvg_direction,
            f"FVG {fvg_direction} active — imbalance created by impulsive move",
            weight=0.65,
        ))
    if bias != "NEUTRAL":
        exp_sigs.append(RegimeSignal(
            "Market structure bias", bias,
            f"Higher-TF market structure is {bias} — trend continuation favoured",
            weight=0.5,
        ))

    # --- MANIPULATION signals ---
    if liquidity_sweep:
        strength_label = ("STRONG" if sweep_strength > 0.7
                          else "MODERATE" if sweep_strength > 0.4 else "WEAK")
        man_sigs.append(RegimeSignal(
            "Liquidity sweep", sweep_direction,
            f"{strength_label} {sweep_direction} liquidity sweep detected — stop hunt",
            weight=min(1.0, 0.5 + sweep_strength * 0.5),
        ))
    if sweep_reject:
        man_sigs.append(RegimeSignal(
            "Sweep rejection", sweep_direction,
            "Price swept liquidity then immediately rejected — manipulation complete",
            weight=0.90,
        ))
    if sweep_confirm:
        man_sigs.append(RegimeSignal(
            "Sweep confirmed", sweep_direction,
            "Sweep confirmed by subsequent price action — expect reversal",
            weight=0.85,
        ))
    if choch_detected:
        man_sigs.append(RegimeSignal(
            "Change of Character", choch_direction,
            f"CHoCH {choch_direction} — internal structure broken after sweep",
            weight=0.80,
        ))
    if in_order_block:
        direction = "BULLISH" if in_bull_ob else "BEARISH"
        man_sigs.append(RegimeSignal(
            "Price in Order Block", direction,
            f"Price trading inside {direction} OB — institutional interest zone",
            weight=0.70,
        ))
    if ob_active and not in_order_block:
        man_sigs.append(RegimeSignal(
            "Order Block nearby", ob_direction,
            f"{ob_direction} OB present — potential manipulation target",
            weight=0.45,
        ))

    # ── Score each regime ──────────────────────────────────────────────────────
    def score(signals: list[RegimeSignal]) -> float:
        if not signals:
            return 0.0
        total_w = sum(s.weight for s in signals)
        return min(1.0, total_w / max(len(signals), 1))

    raw_scores = {
        "CONSOLIDATION": score(con_sigs),
        "EXPANSION":     score(exp_sigs),
        "MANIPULATION":  score(man_sigs),
    }

    # Normalise to sum = 1 (so they read as relative probabilities)
    total = sum(raw_scores.values()) or 1.0
    regime_scores = {k: round(v / total, 3) for k, v in raw_scores.items()}

    dominant_regime = max(regime_scores, key=regime_scores.__getitem__)

    # ── Narrative ─────────────────────────────────────────────────────────────
    narrative, trade_implication = _build_narrative(
        dominant_regime, regime_scores,
        liquidity_sweep, sweep_direction, sweep_rejected=sweep_reject,
        choch_detected=choch_detected, choch_direction=choch_direction,
        bos_detected=bos_detected, bos_direction=bos_direction,
        bias=bias, pd_zone=pd_zone, atr_vs_avg=atr_vs_avg,
        adx=adx, fvg_active=fvg_active, fvg_direction=fvg_direction,
        ob_active=ob_active, ob_direction=ob_direction,
        in_order_block=in_order_block,
    )

    return RegimeReport(
        dominant_regime        = dominant_regime,
        regime_scores          = regime_scores,
        timestamp              = timestamp,
        close                  = close if close else None,
        atr_pips               = atr_pips,
        atr_vs_avg             = atr_vs_avg,
        bias                   = bias,
        pd_zone                = pd_zone,
        trend_strength         = round(trend_strength, 3) if trend_strength else None,
        adx                    = round(adx, 1) if adx else None,
        liquidity_sweep        = liquidity_sweep,
        sweep_direction        = sweep_direction,
        sweep_rejected         = sweep_reject,
        sweep_confirmed        = sweep_confirm,
        choch_detected         = choch_detected,
        choch_direction        = choch_direction,
        bos_detected           = bos_detected,
        bos_direction          = bos_direction,
        fvg_active             = fvg_active,
        fvg_direction          = fvg_direction,
        ob_active              = ob_active,
        ob_direction           = ob_direction,
        in_order_block         = in_order_block,
        ob_bullish_top         = ob_bullish_top,
        ob_bullish_bottom      = ob_bullish_bottom,
        ob_bearish_top         = ob_bearish_top,
        ob_bearish_bottom      = ob_bearish_bottom,
        consolidation_signals  = con_sigs,
        expansion_signals      = exp_sigs,
        manipulation_signals   = man_sigs,
        narrative              = narrative,
        trade_implication      = trade_implication,
    )


def _build_narrative(
    dominant: str, scores: dict,
    liquidity_sweep: bool, sweep_direction: str,
    sweep_rejected: bool, choch_detected: bool, choch_direction: str,
    bos_detected: bool, bos_direction: str,
    bias: str, pd_zone: str, atr_vs_avg: str,
    adx: float, fvg_active: bool, fvg_direction: str,
    ob_active: bool, ob_direction: str, in_order_block: bool,
) -> tuple[str, str]:
    """Build human-readable narrative and trade implication strings."""

    c = scores.get("CONSOLIDATION", 0)
    e = scores.get("EXPANSION",     0)
    m = scores.get("MANIPULATION",  0)

    # Manipulation narrative (highest specificity — ICT model)
    if dominant == "MANIPULATION":
        if sweep_rejected and choch_detected:
            narr = (
                f"MANIPULATION COMPLETE. A {sweep_direction} liquidity sweep was detected "
                f"and subsequently REJECTED — price spiked to take stops then reversed. "
                f"A {choch_direction} Change of Character (CHoCH) has formed, signalling "
                f"that smart money has repositioned."
            )
            impl = (
                f"High-probability {choch_direction} entry setup. Look for the next M15 "
                f"pullback into the CHoCH level or the nearest Order Block as the entry "
                f"trigger. Bias is now {choch_direction}."
            )
        elif liquidity_sweep and not sweep_rejected:
            narr = (
                f"ACTIVE MANIPULATION. A {sweep_direction} liquidity sweep is in progress "
                f"— price is hunting stops below/above recent swing points. "
                f"The sweep has NOT been rejected yet; waiting for rejection candle."
            )
            impl = (
                f"DO NOT chase the sweep direction. Wait for a strong rejection "
                f"(bearish/bullish engulfing) and CHoCH confirmation before entering "
                f"in the counter-direction."
            )
        elif in_order_block:
            narr = (
                f"POTENTIAL MANIPULATION. Price is trading inside an {ob_direction} "
                f"Order Block — a zone where institutional orders are likely resting. "
                f"Watch for a wick/rejection candle as confirmation of manipulation."
            )
            impl = (
                f"Monitor price reaction inside the OB. A strong close outside the OB "
                f"(expansion) or a rejection wick (manipulation complete) will confirm "
                f"the next move."
            )
        else:
            narr  = "POTENTIAL MANIPULATION. Institutional activity signs present."
            impl  = "Wait for sweep rejection and CHoCH before committing to a direction."

    # Expansion narrative
    elif dominant == "EXPANSION":
        if bos_detected:
            narr = (
                f"EXPANSION CONFIRMED. A {bos_direction} Break of Structure (BOS) has "
                f"occurred — price has closed above/below a prior swing point, confirming "
                f"a new leg. ATR is {atr_vs_avg.lower()}. Market structure bias: {bias}."
            )
            impl = (
                f"Trend is {bos_direction}. Enter on pullbacks to the nearest {bos_direction} "
                f"Order Block or Fair Value Gap. Do not sell into {bos_direction} BOS."
            )
        elif adx >= ADX_STRONG_THRESHOLD:
            narr = (
                f"STRONG EXPANSION. ADX={adx:.1f} — a powerful trend is in motion. "
                f"Market structure: {bias}. ATR: {atr_vs_avg.lower()}."
            )
            impl = "Trade in the trend direction only. Pullback entries to OBs are safest."
        else:
            narr = (
                f"MODERATE EXPANSION. Directional move detected. Bias: {bias}. "
                f"ATR: {atr_vs_avg.lower()}. Trend building but not yet confirmed by BOS."
            )
            impl = "Wait for BOS or FVG fill as entry confirmation before committing."

    # Consolidation narrative
    else:
        if atr_vs_avg == "COMPRESSED" or pd_zone == "EQUILIBRIUM":
            narr = (
                f"CONSOLIDATION (Accumulation/Distribution). Price is coiling — "
                f"ATR is {atr_vs_avg.lower()}, market is at {pd_zone}. "
                f"ADX={adx:.1f} confirms no directional trend. "
                f"Smart money is building a position before the next expansion."
            )
            impl = (
                f"Do NOT trade inside consolidation. Wait for a liquidity sweep of the "
                f"range high or low followed by a BOS in the opposite direction. "
                f"The breakout direction will be opposite to the sweep (ICT model)."
            )
        else:
            narr = (
                f"CONSOLIDATION. Price is ranging with no clear trend (ADX={adx:.1f}). "
                f"Market is at {pd_zone} of the current swing range."
            )
            impl = "Avoid new entries until a clear BOS or manipulation sequence forms."

    # Add FVG/OB context
    if fvg_active:
        impl += f" Active {fvg_direction} FVG present — a pullback to fill it is possible before the next leg."
    if ob_active and not in_order_block:
        impl += f" Nearest {ob_direction} Order Block is a key area to watch for reaction."

    return narr, impl


# ── Console printer ───────────────────────────────────────────────────────────

def print_regime_report(report: RegimeReport) -> None:
    """Print a formatted regime report to stdout."""
    sep   = "=" * 65
    line  = "-" * 65
    EMOJI = {
        "CONSOLIDATION": "[CON]",
        "EXPANSION":     "[EXP]",
        "MANIPULATION":  "[MAN]",
        "BULLISH":       "[BULL]",
        "BEARISH":       "[BEAR]",
        "NEUTRAL":       "[NEUT]",
    }

    print(sep)
    print("  MARKET REGIME ANALYSIS — EURUSD M15")
    print(f"  {report.timestamp}  |  close={report.close:.5f}" if report.close else f"  {report.timestamp}")
    print(sep)

    # Regime scores bar
    for regime, score in sorted(report.regime_scores.items(), key=lambda x: -x[1]):
        bar_len = int(score * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        marker  = " <-- DOMINANT" if regime == report.dominant_regime else ""
        print(f"  {regime:15s}  {bar}  {score:.0%}{marker}")

    print(line)

    # Context table
    rows = [
        ("ATR",          f"{report.atr_pips} pips  [{report.atr_vs_avg}]" if report.atr_pips else "N/A"),
        ("ADX",          f"{report.adx}" if report.adx is not None else "N/A"),
        ("Bias",         f"{report.bias}"),
        ("P/D Zone",     f"{report.pd_zone}"),
        ("Trend strength", f"{report.trend_strength}" if report.trend_strength is not None else "N/A"),
    ]
    for label, value in rows:
        print(f"  {label:16s}  {value}")

    print(line)
    print("  ICT / SMC Signals")
    print(line)

    ict_rows = [
        ("Liquidity sweep",   f"{report.sweep_direction}" if report.liquidity_sweep else "None"),
        ("Sweep rejected",    "YES" if report.sweep_rejected else "No"),
        ("Sweep confirmed",   "YES" if report.sweep_confirmed else "No"),
        ("CHoCH",             f"{report.choch_direction}" if report.choch_detected else "None"),
        ("BOS",               f"{report.bos_direction}" if report.bos_detected else "None"),
        ("FVG active",        f"{report.fvg_direction}" if report.fvg_active else "None"),
        ("OB active",         f"{report.ob_direction}" if report.ob_active else "None"),
        ("Price in OB",       "YES" if report.in_order_block else "No"),
    ]
    for label, value in ict_rows:
        print(f"  {label:18s}  {value}")

    print(line)
    print("  What's happening:")
    for word in report.narrative.split(". "):
        if word.strip():
            print(f"    {word.strip()}.")
    print()
    print("  What to do:")
    for word in report.trade_implication.split(". "):
        if word.strip():
            print(f"    {word.strip()}.")
    print(sep)

    # Evidence breakdown
    if report.consolidation_signals:
        print(f"\n  CONSOLIDATION evidence ({len(report.consolidation_signals)} signals):")
        for s in report.consolidation_signals:
            print(f"    [{s.weight:.0%}] {s.name}: {s.description}")
    if report.expansion_signals:
        print(f"\n  EXPANSION evidence ({len(report.expansion_signals)} signals):")
        for s in report.expansion_signals:
            print(f"    [{s.weight:.0%}] {s.name}: {s.description}")
    if report.manipulation_signals:
        print(f"\n  MANIPULATION evidence ({len(report.manipulation_signals)} signals):")
        for s in report.manipulation_signals:
            print(f"    [{s.weight:.0%}] {s.name}: {s.description}")
    print()
