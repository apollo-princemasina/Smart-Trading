"""DashboardService — single-call aggregation of all engine outputs."""
from __future__ import annotations

import time
from typing import Any


class DashboardService:
    """
    Reads live state from every Phase 1–5 engine and the DB to produce
    one complete dashboard snapshot.

    This service contains NO trading logic — it is a read-only aggregator.
    """

    def __init__(self, app_state: Any, session_factory, start_time: float) -> None:
        self._state = app_state
        self._session_factory = session_factory
        self._start_time = start_time

    async def snapshot(self) -> dict:
        return {
            "decision":        await self._decision(),
            "latest_prediction": await self._latest_prediction(),
            "market_regime":   self._regime(),
            "mia_summary":     self._mia_summary(),
            "eie_summary":     self._eie_summary(),
            "buffer_status":   self._buffer_status(),
            "system_summary":  self._system_summary(),
        }

    # ── Private collectors ────────────────────────────────────────────────

    async def _decision(self) -> dict | None:
        try:
            from decision_fusion.recommendation_cache.cache import decision_cache
            d = decision_cache.current
            if d is None:
                return None
            exp_min = None
            if d.expires_at and d.generated_at:
                try:
                    exp_min = round((d.expires_at - d.generated_at).total_seconds() / 60, 1)
                except Exception:
                    pass
            return {
                "recommendation":        str(d.recommendation).replace("Recommendation.", ""),
                "strength":              str(d.recommendation_strength).replace("RecommendationStrength.", ""),
                "confidence":            d.decision_confidence,
                "agreement_score":       d.agreement_score,
                "conflict_score":        d.conflict_score,
                "consensus_level":       str(d.consensus_level).replace("ConsensusLevel.", ""),
                "market_bias":           str(d.market_bias).replace("MarketBiasEnum.", ""),
                "technical_alignment":   getattr(d, "technical_alignment", 0.0),
                "fundamental_alignment": getattr(d, "fundamental_alignment", 0.0),
                "has_ml":                bool(getattr(d, "has_ml", False)),
                "has_eie":               bool(getattr(d, "has_eie", False)),
                "has_mia":               bool(getattr(d, "has_mia", False)),
                "primary_reasons":       list(d.primary_reasons[:4]),
                "conflicting_factors":   list(getattr(d, "conflicting_factors", [])[:3]),
                "risk_factors":          list(d.risk_factors[:3]),
                "generated_at":          d.generated_at.isoformat() if d.generated_at else None,
                "expires_at":            d.expires_at.isoformat() if d.expires_at else None,
                "expiry_minutes":        exp_min,
                "is_expired":            decision_cache.is_expired(),
                "age_seconds":           decision_cache.age_seconds(),
                "schema_version":        d.decision_schema_version,
            }
        except Exception:
            return None

    async def _latest_prediction(self) -> dict | None:
        try:
            async with self._session_factory() as session:
                from src.database.repositories.prediction_repo import PredictionRepository
                repo = PredictionRepository(session)
                p = await repo.latest()
                if p is None:
                    return None
                meta = p.metadata_json or {}
                return {
                    "id":            str(p.id),
                    "direction":     p.direction,
                    "raw_direction": meta.get("raw_direction"),
                    "demoted":       meta.get("demoted", False),
                    "confidence":    p.confidence,
                    "raw_confidence": p.raw_confidence,
                    "prob_buy":      p.prob_buy,
                    "prob_sell":     p.prob_sell,
                    "prob_hold":     p.prob_hold,
                    "regime":        p.regime,
                    "close":         p.close,
                    "tp_price":      p.tp_price,
                    "sl_price":      p.sl_price,
                    "tp_pips":       p.tp_pips,
                    "sl_pips":       p.sl_pips,
                    "atr_pips":      p.atr_pips,
                    "session":       p.session,
                    "session_mult":  p.session_mult,
                    "signal_time":   p.signal_time.isoformat() if p.signal_time else None,
                }
        except Exception:
            return None

    def _regime(self) -> dict | None:
        try:
            ie = self._state.inference_engine
            regime = ie.latest_regime()
            if regime is None:
                return None
            raw = regime.regime_scores or {}
            return {
                "dominant": regime.dominant_regime,
                "scores": {
                    "consolidation": raw.get("CONSOLIDATION", 0.0),
                    "expansion":     raw.get("EXPANSION", 0.0),
                    "manipulation":  raw.get("MANIPULATION", 0.0),
                },
                "bias":      regime.bias,
                "pd_zone":   regime.pd_zone,
                "atr_pips":  regime.atr_pips,
                "adx":       regime.adx,
                "narrative": regime.narrative,
                "trade_impl": regime.trade_implication,
                "ict": {
                    "liquidity_sweep":  regime.liquidity_sweep,
                    "sweep_direction":  regime.sweep_direction,
                    "choch_detected":   regime.choch_detected,
                    "bos_detected":     regime.bos_detected,
                    "fvg_active":       regime.fvg_active,
                    "ob_active":        regime.ob_active,
                },
            }
        except Exception:
            return None

    def _mia_summary(self) -> dict | None:
        try:
            mia = self._state.mia
            out = mia.latest_analysis
            if out is None:
                return None
            return {
                "market_bias":       str(getattr(out, "market_bias", "")).replace("MarketBias.", ""),
                "confidence":        getattr(out, "confidence", None),
                "risk_level":        str(getattr(out, "risk_level", "")).replace("RiskLevel.", ""),
                "market_summary":    getattr(out, "market_summary", None),
                "expected_duration": str(getattr(out, "expected_duration", "")).replace("TimeHorizon.", ""),
                "is_fallback":       getattr(out, "is_fallback", False),
                "timestamp":         out.timestamp.isoformat() if getattr(out, "timestamp", None) else None,
            }
        except Exception:
            return None

    def _eie_summary(self) -> dict | None:
        import datetime as _dt
        try:
            eie = self._state.eie
            reports = []
            if hasattr(eie, "get_active_reports"):
                reports = eie.get_active_reports() or []

            # Pull next 6 upcoming events from the FF connector cache
            upcoming: list[dict] = []
            try:
                from forex_factory_connector.cache.memory_cache import connector_cache
                week_cache = connector_cache._weeks.get("thisweek")
                if week_cache and week_cache.events:
                    now = _dt.datetime.now(_dt.timezone.utc)
                    future = [
                        e for e in week_cache.events
                        if getattr(e, "timestamp_utc", None)
                        and e.timestamp_utc > now
                    ]
                    future.sort(key=lambda e: e.timestamp_utc)
                    for ev in future[:6]:
                        t = ev.timestamp_utc.strftime("%H:%M UTC")
                        upcoming.append({
                            "title":    ev.title,
                            "currency": ev.currency,
                            "impact":   ev.impact if isinstance(ev.impact, str) else ev.impact.value,
                            "time":     t,
                            "forecast": getattr(ev, "forecast", None),
                            "previous": getattr(ev, "previous", None),
                        })
            except Exception:
                pass

            return {
                "active_count":    len(reports),
                "has_active_events": len(reports) > 0,
                "execution_risk":  max(
                    (getattr(r, "execution_risk", 0) for r in reports), default=0.0
                ),
                "upcoming": upcoming,
            }
        except Exception:
            return None

    def _buffer_status(self) -> dict | None:
        try:
            rb = self._state.rolling_buffer
            return rb.status() if hasattr(rb, "status") else {}
        except Exception:
            return None

    def _system_summary(self) -> dict:
        uptime = round(time.monotonic() - self._start_time, 1)
        ws_count = getattr(
            getattr(self._state, "ws_manager", None), "connection_count", 0
        )
        return {
            "uptime_seconds": uptime,
            "websocket_connections": ws_count,
            "scheduler_running": bool(getattr(
                getattr(self._state, "scheduler", None), "_running", True
            )),
        }
