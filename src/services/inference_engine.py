"""Inference Engine — orchestrates the full feature → predict → regime cycle."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from src.api.core.config import settings

if TYPE_CHECKING:
    from src.inference.market_regime    import RegimeReport
    from src.services.feature_engine    import FeatureEngine
    from src.services.pipeline_manager  import PipelineManager
    from src.services.rolling_buffer    import RollingBufferManager


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
        from src.inference.predictor       import predict
        from src.inference.signal_generator import latest_signal
        from src.inference.market_regime    import analyze_market_regime

        # 1. Pull dataframes from buffer
        m15_df = self._buffer.as_dataframe("M15")
        htf_dfs = {
            tf: self._buffer.as_dataframe(tf)
            for tf in ("H1", "H4", "D1", "W1")
        }

        if m15_df.empty or len(m15_df) < 50:
            logger.warning("Inference skipped — M15 buffer has {} rows", len(m15_df))
            return None

        # 2. Build features (runs in executor — sync pipeline)
        logger.debug("Building features for {} M15 bars...", len(m15_df))
        feature_df = await self._feature_engine.build(m15_df, htf_dfs)

        # 3. Predict
        preds, probas = predict(
            feature_df,
            bundle_dir=self._pipeline.bundle_dir,
        )

        # 4. Generate signal for the latest bar
        sig = latest_signal(
            preds, probas, feature_df,
            min_confidence=settings.INFERENCE_MIN_CONFIDENCE,
            directional_only=True,
        )

        # 5. Regime analysis (always computed, even without a signal)
        regime = analyze_market_regime(feature_df)
        self._last_regime = regime

        if sig is None:
            logger.debug("No qualifying signal this cycle")
            self._last_result = None
            return None

        # 6. Session-aware confidence weighting
        from src.services.session_weighting import get_session, apply_session_weighting
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
        if adj_conf < settings.INFERENCE_MIN_CONFIDENCE and sig.direction != "HOLD":
            logger.info(
                "Signal demoted to HOLD: {} {:.0%} raw -> {:.0%} after {} weighting (x{:.2f})",
                sig.direction, raw_confidence, adj_conf,
                session.name, session.multiplier,
            )
            self._last_result = None
            return None

        effective_direction = sig.direction

        # 7. TP / SL calculation (use raw ATR — session weighting doesn't change levels)
        atr       = sig.atr or 0.0008
        atr_pips  = round(atr / 0.0001, 1)
        tp_mult   = settings.INFERENCE_TP_ATR_MULT
        sl_mult   = settings.INFERENCE_SL_ATR_MULT

        if effective_direction == "BUY":
            tp_price = sig.close + atr * tp_mult
            sl_price = sig.close - atr * sl_mult
        else:
            tp_price = sig.close - atr * tp_mult
            sl_price = sig.close + atr * sl_mult

        result: dict[str, Any] = {
            "signal_time":       sig.timestamp,
            "symbol":            settings.MODEL_SYMBOL,
            "timeframe":         "M15",
            "direction":         effective_direction,
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
            "model_version":     self._pipeline.model_name,
            "created_at":        datetime.now(timezone.utc),
        }

        self._last_result = result
        logger.info(
            "Signal: {} {:.0%} conf ({:.0%} raw x{:.2f} {})  close={:.5f}  TP+{} SL-{} pips  regime={}",
            result["direction"], result["confidence"], raw_confidence,
            session.multiplier, session.name, result["close"],
            result["tp_pips"], result["sl_pips"], result["regime"],
        )
        return result
