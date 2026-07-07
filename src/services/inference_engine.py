"""Inference Engine — orchestrates the full feature → predict → regime cycle. v2"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from src.api.core.config import settings

if TYPE_CHECKING:
    from src.inference.market_regime    import RegimeReport
    from src.services.feature_engine    import FeatureEngine
    from src.services.pipeline_manager  import PipelineManager
    from src.services.rolling_buffer    import RollingBufferManager

_ROOT = Path(__file__).resolve().parents[2]

# Conviction level constants
_CONVICTION_HIGH       = "HIGH_CONVICTION"    # 1b + 4b + 8b all agree on same direction
_CONVICTION_SETUP      = "SETUP_FORMING"      # 4b + 8b agree, 1b is HOLD (structural, not yet triggered)
_CONVICTION_BIAS       = "DIRECTIONAL_BIAS"   # at least one lookahead model is directional
_CONVICTION_CONFLICTED = "CONFLICTED"         # lookahead models disagree with each other
_CONVICTION_NEUTRAL    = "NEUTRAL"            # all models say HOLD


def _compute_conviction(
    dir_1b: str,
    dir_4b: str,
    dir_8b: str,
    prob_buy_1b: float,
    prob_sell_1b: float,
) -> dict[str, Any]:
    """Combine 1b/4b/8b directions into a conviction assessment."""
    directional_4b = dir_4b in ("BUY", "SELL")
    directional_8b = dir_8b in ("BUY", "SELL")
    directional_1b = dir_1b in ("BUY", "SELL")

    if directional_4b and directional_8b and dir_4b == dir_8b:
        structural_dir = dir_4b
        if directional_1b and dir_1b == structural_dir:
            level = _CONVICTION_HIGH
            desc  = f"All 3 models agree: {structural_dir} — high-probability structural setup"
        elif not directional_1b:
            level = _CONVICTION_SETUP
            desc  = (
                f"Structural {structural_dir} forming (4h + 2h aligned) — "
                "15-min bar consolidating, wait for entry trigger"
            )
        else:
            level = _CONVICTION_CONFLICTED
            desc  = (
                f"4b+8b say {structural_dir}, 1b says {dir_1b} — "
                "structural and immediate directions conflict"
            )
    elif directional_4b or directional_8b:
        dirs  = [d for d in (dir_4b, dir_8b) if d in ("BUY", "SELL")]
        level = _CONVICTION_BIAS
        desc  = f"Directional bias: {', '.join(dirs)} — single lookahead agreement, low conviction"
    elif directional_1b:
        level = _CONVICTION_BIAS
        desc  = f"Short-term {dir_1b} signal only — no structural confirmation from 4b/8b"
    else:
        level = _CONVICTION_NEUTRAL
        desc  = "All models neutral — no directional edge identified"

    return {"level": level, "structural_dir": dir_4b if dir_4b == dir_8b else None, "description": desc}


class InferenceEngine:
    """Runs one complete M15 inference cycle and returns a structured result."""

    def __init__(
        self,
        pipeline_manager: "PipelineManager",
        rolling_buffer:   "RollingBufferManager",
    ) -> None:
        self._pipeline    = pipeline_manager
        self._buffer      = rolling_buffer
        self._feat_engine: Optional["FeatureEngine"] = None
        self._last_regime: Optional["RegimeReport"]  = None
        self._last_result: Optional[dict[str, Any]]  = None

        from src.services.ict_state_machine import ICTEntryStateMachine
        self._ict_sm = ICTEntryStateMachine()

        # Lookahead model bundle dirs — loaded only if present
        self._bundle_4b: Optional[Path] = None
        self._bundle_8b: Optional[Path] = None
        b4 = _ROOT / "models" / "lookahead_4b"
        b8 = _ROOT / "models" / "lookahead_8b"
        if (b4 / "model.joblib").exists():
            self._bundle_4b = b4
            logger.info("InferenceEngine: lookahead_4b model loaded ({})", b4)
        else:
            logger.warning("InferenceEngine: lookahead_4b bundle not found — conviction scoring disabled")
        if (b8 / "model.joblib").exists():
            self._bundle_8b = b8
            logger.info("InferenceEngine: lookahead_8b model loaded ({})", b8)
        else:
            logger.warning("InferenceEngine: lookahead_8b bundle not found — conviction scoring disabled")

    # Lazy-init the FeatureEngine (avoids circular import at module load)
    @property
    def _feature_engine(self) -> "FeatureEngine":
        if self._feat_engine is None:
            from src.services.feature_engine import FeatureEngine
            self._feat_engine = FeatureEngine()
        return self._feat_engine

    # ── Public interface ──────────────────────────────────────────────────────

    async def run_inference(self) -> Optional[dict[str, Any]]:
        """Run a full inference cycle on the current buffer state.

        Returns a dict with signal + regime data, or None if:
          - Buffer not ready
          - No qualifying signal (HOLD or below confidence threshold)
        """
        if not self._buffer.all_ready:
            logger.warning("Inference skipped — buffers not fully ready")
            return None

        try:
            return await self._run()
        except Exception:
            logger.exception("Inference cycle failed")
            return None

    def latest_regime(self) -> Optional["RegimeReport"]:
        return self._last_regime

    def latest_result(self) -> Optional[dict[str, Any]]:
        return self._last_result

    # ── Core cycle ────────────────────────────────────────────────────────────

    async def _run(self) -> Optional[dict[str, Any]]:
        m15_df = self._buffer.as_dataframe("M15")
        htf_dfs = {
            tf: self._buffer.as_dataframe(tf)
            for tf in ("H1", "H4", "D1", "W1")
        }
        if m15_df.empty or len(m15_df) < 50:
            logger.warning("Inference skipped — M15 buffer has {} rows", len(m15_df))
            return None
        return await self._run_on_data(m15_df, htf_dfs)

    async def run_inference_at(
        self,
        m15_df: Any,
        htf_dfs: Any,
    ) -> Optional[dict[str, Any]]:
        """Run inference on a specific historical bar slice (for startup catch-up).

        The target bar is always the last row of m15_df.  Session detection uses
        the bar's own timestamp + 1 min (≈ when the cron would have fired) so that
        historical bars are assessed against the session that was active at that time,
        not the current wall-clock session.
        """
        if m15_df is None or len(m15_df) < 50:
            logger.warning("run_inference_at: insufficient data ({} rows)", len(m15_df) if m15_df is not None else 0)
            return None
        try:
            return await self._run_on_data(m15_df, htf_dfs, use_bar_session=True)
        except Exception:
            logger.exception("run_inference_at failed")
            return None

    async def _run_on_data(
        self,
        m15_df: Any,
        htf_dfs: Any,
        *,
        use_bar_session: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Core inference computation.  m15_df last row is the target bar."""
        from datetime import timedelta

        from src.inference.predictor        import predict
        from src.inference.signal_generator import latest_signal
        from src.inference.market_regime    import analyze_market_regime

        # 1. Build features (runs in executor — sync pipeline)
        logger.debug("Building features for {} M15 bars...", len(m15_df))
        feature_df = await self._feature_engine.build(m15_df, htf_dfs)

        # 2. Predict — primary 1b model
        preds, probas = predict(
            feature_df,
            bundle_dir=self._pipeline.bundle_dir,
        )

        # 2b. Lookahead models (4b = 1h, 8b = 2h) — run in parallel when available
        preds_4b = probas_4b = None
        preds_8b = probas_8b = None
        if self._bundle_4b is not None:
            try:
                preds_4b, probas_4b = predict(feature_df, bundle_dir=self._bundle_4b)
            except Exception:
                logger.warning("lookahead_4b inference failed — skipping conviction")
        if self._bundle_8b is not None:
            try:
                preds_8b, probas_8b = predict(feature_df, bundle_dir=self._bundle_8b)
            except Exception:
                logger.warning("lookahead_8b inference failed — skipping conviction")

        # 3. Get signal for the current (last) bar — no confidence/direction filter here.
        # Session weighting and the demote logic below handle thresholds.
        # Using min_confidence=0 + directional_only=False ensures all_signals[-1]
        # is always the final row, not an old bar that happened to be the last to
        # exceed the threshold.
        sig = latest_signal(
            preds, probas, feature_df,
            min_confidence=0.0,
            directional_only=False,
        )

        # 4. Regime analysis (always computed, even without a signal)
        regime = analyze_market_regime(feature_df)
        self._last_regime = regime

        if sig is None:
            logger.debug("No qualifying signal this cycle")
            self._last_result = None
            return None

        # 5. Session-aware confidence weighting.
        # For catch-up bars, use the bar's own open+1 min as the session reference
        # so the historical session is accurate.  For live bars, use current time.
        from src.services.session_weighting import get_session, apply_session_weighting
        if use_bar_session and sig.timestamp is not None:
            try:
                fire_dt   = sig.timestamp + timedelta(minutes=1)
                fire_hour = fire_dt.hour
            except Exception:
                fire_hour = None
            session = get_session(utc_hour=fire_hour)
        else:
            session = get_session()

        # Market closed on weekends — skip all signal generation
        if session.name == "MARKET_CLOSED":
            logger.info("Inference skipped — forex market closed (weekend)")
            self._last_result = None
            return None

        raw_confidence = round(sig.confidence, 4)

        adj_sell, adj_hold, adj_buy, adj_conf = apply_session_weighting(
            sig.prob_sell, sig.prob_hold, sig.prob_buy,
            sig.direction, session,
        )

        # If weighting pushes us below threshold, demote to HOLD — no directional trade
        demoted = False
        if adj_conf < settings.INFERENCE_MIN_CONFIDENCE and sig.direction != "HOLD":
            logger.info(
                "Signal demoted to HOLD: {} {:.0%} raw -> {:.0%} after {} weighting (x{:.2f})",
                sig.direction, raw_confidence, adj_conf,
                session.name, session.multiplier,
            )
            demoted = True

        effective_direction = "HOLD" if demoted else sig.direction

        # 6. Build lookahead conviction data (last bar of each model)
        CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
        conviction: dict[str, Any] = {}
        setup_forming_alert: Optional[str] = None   # non-None when SETUP_FORMING is active
        conviction_gate_applied = False

        if probas_4b is not None and probas_8b is not None:
            p4 = probas_4b[-1]   # [sell, hold, buy]
            p8 = probas_8b[-1]
            dir_4b = CLASS_NAMES[int(preds_4b[-1])]
            dir_8b = CLASS_NAMES[int(preds_8b[-1])]
            conviction = _compute_conviction(
                effective_direction, dir_4b, dir_8b,
                adj_buy, adj_sell,
            )
            conviction.update({
                "direction_4b":  dir_4b,
                "direction_8b":  dir_8b,
                "prob_buy_4b":   round(float(p4[2]), 4),
                "prob_sell_4b":  round(float(p4[0]), 4),
                "prob_hold_4b":  round(float(p4[1]), 4),
                "prob_buy_8b":   round(float(p8[2]), 4),
                "prob_sell_8b":  round(float(p8[0]), 4),
                "prob_hold_8b":  round(float(p8[1]), 4),
            })
            logger.info(
                "Conviction: {}  1b={} 4b={} 8b={}",
                conviction["level"], effective_direction, dir_4b, dir_8b,
            )

            # Strategy B gate — only fire when all 3 models agree (HIGH_CONVICTION).
            # SETUP_FORMING surfaces as a dashboard alert but does not trigger a trade.
            if settings.INFERENCE_REQUIRE_CONVICTION and effective_direction != "HOLD":
                level = conviction.get("level", _CONVICTION_NEUTRAL)
                if level == _CONVICTION_HIGH:
                    pass   # all 3 agree — proceed normally
                elif level == _CONVICTION_SETUP:
                    # Structural setup forming but 1b hasn't confirmed yet.
                    # Demote direction to HOLD; expose setup direction as alert.
                    setup_forming_alert = conviction.get("structural_dir")
                    effective_direction  = "HOLD"
                    conviction_gate_applied = True
                    logger.info(
                        "Strategy B gate: SETUP_FORMING {} — demoted to HOLD (alert only)",
                        setup_forming_alert,
                    )
                else:
                    # CONFLICTED, DIRECTIONAL_BIAS, or NEUTRAL — no trade
                    conviction_gate_applied = True
                    effective_direction = "HOLD"
                    logger.info(
                        "Strategy B gate: {} — demoted to HOLD (no multi-model agreement)",
                        level,
                    )

        # 7. ICT Entry State Machine — runs every bar to track OB retracement setups.
        #    structural_dir is the 4b+8b consensus (None if no consensus yet).
        structural_dir: Optional[str] = conviction.get("structural_dir") if conviction else None
        ict_entry = self._ict_sm.update(regime, structural_dir, effective_direction)

        ict_ob_entry = False
        if ict_entry is not None:
            # OB retracement entry confirmed — override direction and use precise OB-based SL
            logger.info(
                "ICT OB Entry confirmed: {} SL={:.5f}  bars_waited={}",
                ict_entry.direction, ict_entry.sl_price, ict_entry.bars_waited,
            )
            effective_direction = ict_entry.direction
            ict_ob_entry = True
            # Also lift conviction gate — ICT OB entry IS the gate
            conviction_gate_applied = False

        # 8. TP / SL calculation — uses final gated direction
        atr       = sig.atr or 0.0008
        atr_pips  = round(atr / 0.0001, 1)
        tp_mult   = settings.INFERENCE_TP_ATR_MULT
        sl_mult   = settings.INFERENCE_SL_ATR_MULT

        if ict_entry is not None:
            # Precise SL at OB edge; TP still ATR-based from current close
            sl_price = ict_entry.sl_price
            if effective_direction == "BUY":
                tp_price = sig.close + atr * tp_mult
            else:
                tp_price = sig.close - atr * tp_mult
        elif effective_direction == "BUY":
            tp_price = sig.close + atr * tp_mult
            sl_price = sig.close - atr * sl_mult
        elif effective_direction == "SELL":
            tp_price = sig.close - atr * tp_mult
            sl_price = sig.close + atr * sl_mult
        else:
            tp_price = sig.close - atr * tp_mult
            sl_price = sig.close + atr * sl_mult

        result: dict[str, Any] = {
            "signal_time":       sig.timestamp,
            "symbol":            settings.MODEL_SYMBOL,
            "timeframe":         "M15",
            "direction":         effective_direction,
            "raw_direction":     sig.direction,      # original model direction (before demotion)
            "demoted":           demoted,
            "confidence":        adj_conf,           # session-adjusted
            "raw_confidence":    raw_confidence,     # original model output
            "prob_sell":         adj_sell,
            "prob_hold":         adj_hold,
            "prob_buy":          adj_buy,
            "close":             sig.close,
            "atr_pips":          atr_pips,
            "tp_price":          round(tp_price, 5),
            "sl_price":          round(sl_price, 5),
            "tp_pips":           round(abs(tp_price - sig.close) / 0.0001, 1),
            "sl_pips":           round(abs(sl_price - sig.close) / 0.0001, 1),
            "regime":            regime.dominant_regime,
            "regime_scores":     regime.regime_scores,
            "session":           session.name,
            "session_mult":      session.multiplier,
            "model_version":          self._pipeline.model_name,
            "created_at":             datetime.now(timezone.utc),
            "conviction":             conviction or None,
            "conviction_gate_applied": conviction_gate_applied,
            "setup_forming_alert":    setup_forming_alert,   # directional alert when SETUP_FORMING fired
            # ICT OB entry fields
            "ict_ob_entry":           ict_ob_entry,
            "ict_sm_state":           self._ict_sm.state,
            "ict_sm_direction":       self._ict_sm.armed_direction,
            "ob_bullish_top":         regime.ob_bullish_top,
            "ob_bullish_bottom":      regime.ob_bullish_bottom,
            "ob_bearish_top":         regime.ob_bearish_top,
            "ob_bearish_bottom":      regime.ob_bearish_bottom,
        }

        self._last_result = result
        logger.info(
            "Signal: {} {:.0%} conf ({:.0%} raw x{:.2f} {})  close={:.5f}  TP+{} SL-{} pips  regime={}",
            result["direction"], result["confidence"], raw_confidence,
            session.multiplier, session.name, result["close"],
            result["tp_pips"], result["sl_pips"], result["regime"],
        )
        return result
