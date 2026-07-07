"""Prediction Service — inference → persist → WebSocket broadcast."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker

if TYPE_CHECKING:
    from src.api.websocket.manager     import WebSocketManager
    from src.services.inference_engine import InferenceEngine


class PredictionService:

    def __init__(
        self,
        inference_engine: "InferenceEngine",
        session_factory:  async_sessionmaker,
        ws_manager:       "WebSocketManager",
        app_state:        Any = None,
    ) -> None:
        self._engine    = inference_engine
        self._factory   = session_factory
        self._ws        = ws_manager
        self._app_state = app_state   # optional — wired in main.py lifespan

    # ── Called by scheduler on every M15 bar close ────────────────────────────

    async def run_cycle(self, update_buffers: bool = True) -> Optional[dict[str, Any]]:
        """Run one complete inference + persist + broadcast cycle.

        Args:
            update_buffers: If False, skip the buffer refresh step (used for startup
                            inference where buffers were just populated).
        """

        # 1. Update rolling buffers first (skipped on startup to avoid rate-limit hit)
        if update_buffers:
            try:
                await self._engine._buffer.update()
            except Exception:
                logger.exception("Buffer update failed")

        # 2. Run inference
        result = await self._engine.run_inference()

        # Always broadcast the latest regime, even when no directional signal fires
        # (e.g. Dead Zone demotion). This keeps the regime panel alive.
        regime = self._engine.latest_regime()
        if regime:
            try:
                regime_data = self._build_regime_payload(regime)
                await self._ws.broadcast("regime_update", regime_data)
            except Exception:
                logger.exception("WebSocket regime broadcast failed")

        if result is None:
            logger.debug("No signal this cycle — regime broadcast only")
            return None

        # 3. Persist to PostgreSQL
        try:
            prediction = await self._persist(result)
            result["id"] = prediction.id
        except Exception:
            logger.exception("Failed to persist prediction")

        # 4. Broadcast to all WebSocket clients
        try:
            await self._broadcast(result)
        except Exception:
            logger.exception("WebSocket broadcast failed")

        # 5. Trigger Decision Fusion Engine with latest intelligence
        try:
            await self._trigger_dfe(result)
        except Exception:
            logger.exception("DFE trigger failed")

        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    async def _persist(self, result: dict[str, Any]):
        from src.database.models.prediction import Prediction
        from src.database.repositories.prediction_repo import PredictionRepository

        # Skip duplicate: same signal_time already persisted (e.g. from a restart)
        async with self._factory() as session:
            repo = PredictionRepository(session)
            if await repo.exists_for_signal_time(result["signal_time"], result["symbol"]):
                logger.info("Prediction for signal_time={} already exists — skipping duplicate",
                            result["signal_time"])
                # Return a stub so the broadcast still fires
                class _Stub:
                    id = "duplicate-skipped"
                return _Stub()

        pred = Prediction(
            signal_time    = result["signal_time"],
            symbol         = result["symbol"],
            timeframe      = result["timeframe"],
            direction      = result["direction"],
            confidence     = result["confidence"],
            raw_confidence = result.get("raw_confidence"),
            prob_sell      = result["prob_sell"],
            prob_hold      = result["prob_hold"],
            prob_buy       = result["prob_buy"],
            close          = result["close"],
            atr_pips       = result.get("atr_pips"),
            tp_price       = result.get("tp_price"),
            sl_price       = result.get("sl_price"),
            tp_pips        = result.get("tp_pips"),
            sl_pips        = result.get("sl_pips"),
            regime         = result.get("regime"),
            regime_scores  = result.get("regime_scores"),
            session        = result.get("session"),
            session_mult   = result.get("session_mult"),
            model_version  = result.get("model_version"),
            metadata_json  = {
                "raw_direction": result.get("raw_direction"),
                "demoted":       result.get("demoted", False),
            },
        )

        async with self._factory() as session:
            session.add(pred)
            await session.commit()
            await session.refresh(pred)

        logger.info("Prediction persisted  id={}  dir={}  conf={:.0%}",
                    pred.id[:8], pred.direction, pred.confidence)
        return pred

    # ── WebSocket broadcast ───────────────────────────────────────────────────

    def _build_regime_payload(self, regime) -> dict[str, Any]:
        # Lowercase keys to match the frontend RegimeScores interface
        raw = regime.regime_scores
        return {
            "dominant":    regime.dominant_regime,
            "scores": {
                "consolidation": raw.get("CONSOLIDATION", 0.0),
                "expansion":     raw.get("EXPANSION", 0.0),
                "manipulation":  raw.get("MANIPULATION", 0.0),
            },
            "bias":        regime.bias,
            "pd_zone":     regime.pd_zone,
            "atr_pips":    regime.atr_pips,
            "adx":         regime.adx,
            "narrative":   regime.narrative,
            "trade_impl":  regime.trade_implication,
            "ict": {
                "liquidity_sweep":  regime.liquidity_sweep,
                "sweep_direction":  regime.sweep_direction,
                "sweep_rejected":   regime.sweep_rejected,
                "sweep_confirmed":  regime.sweep_confirmed,
                "choch_detected":   regime.choch_detected,
                "choch_direction":  regime.choch_direction,
                "bos_detected":     regime.bos_detected,
                "bos_direction":    regime.bos_direction,
                "fvg_active":       regime.fvg_active,
                "fvg_direction":    regime.fvg_direction,
                "ob_active":        regime.ob_active,
                "ob_direction":     regime.ob_direction,
                "in_order_block":   regime.in_order_block,
            },
        }

    async def _broadcast(self, result: dict[str, Any]) -> None:
        regime = self._engine.latest_regime()
        regime_data = {}
        if regime:
            regime_data = self._build_regime_payload(regime)

        payload = {
            "id":             result.get("id", ""),
            "signal_time":    str(result["signal_time"]),
            "symbol":         result["symbol"],
            "timeframe":      result["timeframe"],
            "direction":      result["direction"],
            "raw_direction":  result.get("raw_direction"),
            "demoted":        result.get("demoted", False),
            "confidence":     result["confidence"],
            "raw_confidence": result.get("raw_confidence"),
            "prob_sell":      result["prob_sell"],
            "prob_hold":      result["prob_hold"],
            "prob_buy":       result["prob_buy"],
            "close":          result["close"],
            "atr_pips":       result.get("atr_pips"),
            "tp_price":       result.get("tp_price"),
            "sl_price":       result.get("sl_price"),
            "tp_pips":        result.get("tp_pips"),
            "sl_pips":        result.get("sl_pips"),
            "session":                result.get("session"),
            "session_mult":           result.get("session_mult"),
            "regime":                 regime_data,
            "conviction":             result.get("conviction"),
            "conviction_gate_applied": result.get("conviction_gate_applied", False),
            "setup_forming_alert":    result.get("setup_forming_alert"),
            # ICT OB entry tracking
            "ict_ob_entry":           result.get("ict_ob_entry", False),
            "ict_sm_state":           result.get("ict_sm_state", "IDLE"),
            "ict_sm_direction":       result.get("ict_sm_direction"),
            "ob_bullish_top":         result.get("ob_bullish_top"),
            "ob_bullish_bottom":      result.get("ob_bullish_bottom"),
            "ob_bearish_top":         result.get("ob_bearish_top"),
            "ob_bearish_bottom":      result.get("ob_bearish_bottom"),
        }

        await self._ws.broadcast("signal_update", payload)
        # regime_update already broadcast in run_cycle() above; skip duplicate
        logger.debug("WebSocket signal broadcast sent  clients={}", self._ws.connection_count)

    async def _trigger_dfe(self, result: dict[str, Any]) -> None:
        """Build a FusionInput from current engine state and run the DFE pipeline."""
        if self._app_state is None:
            return
        dfe = getattr(self._app_state, "dfe", None)
        if dfe is None:
            return

        from decision_fusion.models.fusion_input import FusionInput

        eie_reports: list = []
        try:
            eie = getattr(self._app_state, "eie", None)
            if eie and hasattr(eie, "get_active_reports"):
                eie_reports = eie.get_active_reports() or []
        except Exception:
            pass

        mia_output = None
        try:
            mia = getattr(self._app_state, "mia", None)
            if mia:
                mia_output = getattr(mia, "latest_analysis", None)
        except Exception:
            pass

        rb = getattr(self._app_state, "rolling_buffer", None)
        buffer_status = rb.status() if rb and hasattr(rb, "status") else {}

        fi = FusionInput(
            ml_prediction  = result,
            eie_reports    = eie_reports,
            mia_output     = mia_output,
            latest_close   = result.get("close"),
            buffer_ready   = True,
            buffer_status  = buffer_status,
        )

        decision = await dfe.process(fi)

        # Broadcast DFE decision via WebSocket
        try:
            from decision_fusion.models.enums import Recommendation
            payload = {
                "recommendation":  str(decision.recommendation).replace("Recommendation.", ""),
                "strength":        str(decision.recommendation_strength).replace("RecommendationStrength.", ""),
                "confidence":      decision.decision_confidence,
                "agreement_score": decision.agreement_score,
                "conflict_score":  decision.conflict_score,
                "consensus_level": str(decision.consensus_level).replace("ConsensusLevel.", ""),
                "market_bias":           str(decision.market_bias).replace("MarketBiasEnum.", ""),
                "technical_alignment":   getattr(decision, "technical_alignment", None),
                "fundamental_alignment": getattr(decision, "fundamental_alignment", None),
                "primary_reasons":       decision.primary_reasons[:3],
                "conflicting_factors":   getattr(decision, "conflicting_factors", [])[:3],
                "risk_factors":          decision.risk_factors[:3],
                "expiry_minutes":        getattr(decision, "expiry_minutes", None),
                "has_ml":                True,
                "has_eie":               len(eie_reports) > 0,
                "has_mia":               mia_output is not None,
                "generated_at":          decision.generated_at.isoformat() if decision.generated_at else None,
            }
            await self._ws.broadcast("decision_update", payload)
            logger.info("DFE decision broadcast: {} {}", payload["recommendation"], payload["strength"])
        except Exception:
            logger.exception("DFE decision WS broadcast failed")
