"""
Evidence Collector — normalizes all upstream intelligence into EvidenceItems.

Each subsystem is treated as an independent source of evidence.
No averaging, voting, or weighting occurs here — only normalization.
The Agreement, Confidence, and Rule Engines receive the raw evidence list.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from decision_fusion.models.enums import EvidenceDirection, SourceType
from decision_fusion.models.evidence import EvidenceItem
from decision_fusion.models.fusion_input import FusionInput
from decision_fusion.utils.config import dfe_config
from decision_fusion.utils.logger import logger


# ── Direction mapping helpers ──────────────────────────────────────────────────

_ML_DIRECTION_MAP: dict[str, EvidenceDirection] = {
    "BUY":  EvidenceDirection.BULLISH,
    "SELL": EvidenceDirection.BEARISH,
    "HOLD": EvidenceDirection.NEUTRAL,
}

_EIE_DIRECTION_MAP: dict[str, EvidenceDirection] = {
    "BULLISH":   EvidenceDirection.BULLISH,
    "BEARISH":   EvidenceDirection.BEARISH,
    "NEUTRAL":   EvidenceDirection.NEUTRAL,
    "UNCERTAIN": EvidenceDirection.UNCERTAIN,
}

_MIA_BIAS_MAP: dict[str, EvidenceDirection] = {
    "BULLISH":   EvidenceDirection.BULLISH,
    "BEARISH":   EvidenceDirection.BEARISH,
    "NEUTRAL":   EvidenceDirection.NEUTRAL,
    "UNCERTAIN": EvidenceDirection.UNCERTAIN,
}

_IMPORTANCE_MAP: dict[str, float] = {
    "HIGH":   1.0,
    "MEDIUM": 0.7,
    "LOW":    0.4,
}


class EvidenceCollector:
    """
    Collects evidence from every upstream subsystem and normalizes it into
    a canonical list of EvidenceItems.

    This is the entry point for all intelligence into the Decision Fusion Engine.
    """

    def collect(self, fusion_input: FusionInput) -> List[EvidenceItem]:
        """
        Collect and normalize all available evidence.

        Returns a list of EvidenceItems, one per active source.
        Sources that are unavailable produce an ABSENT item so downstream
        components know the source was considered but had no data.
        """
        items: List[EvidenceItem] = []

        items.extend(self._collect_ml(fusion_input))
        items.extend(self._collect_eie(fusion_input))
        items.extend(self._collect_mia(fusion_input))

        logger.debug(
            "Evidence collected: {} items ({} directional)",
            len(items),
            sum(1 for i in items if i.is_directional),
        )
        return items

    # ── ML Evidence ───────────────────────────────────────────────────────────

    def _collect_ml(self, fi: FusionInput) -> List[EvidenceItem]:
        if fi.ml_prediction is None:
            return [self._absent_item(
                SourceType.TECHNICAL_ML, "ML — no prediction available"
            )]

        raw_direction = fi.ml_direction or "HOLD"
        direction = _ML_DIRECTION_MAP.get(raw_direction.upper(), EvidenceDirection.NEUTRAL)
        confidence = max(0.0, min(1.0, fi.ml_confidence or 0.0))

        metadata = {
            "prob_buy":      fi.ml_prediction.get("prob_buy"),
            "prob_sell":     fi.ml_prediction.get("prob_sell"),
            "prob_hold":     fi.ml_prediction.get("prob_hold"),
            "raw_confidence": fi.ml_prediction.get("raw_confidence"),
            "regime":        fi.ml_prediction.get("regime"),
            "session":       fi.ml_prediction.get("session"),
            "session_mult":  fi.ml_prediction.get("session_mult"),
            "model_version": fi.ml_prediction.get("model_version"),
        }

        item = EvidenceItem(
            source      = SourceType.TECHNICAL_ML,
            direction   = direction,
            confidence  = confidence,
            reliability = dfe_config.DFE_RELIABILITY_ML,
            importance  = 1.0,  # Always the primary signal
            timestamp   = fi.ml_prediction.get("signal_time") or fi.current_time,
            label       = f"ML {raw_direction} ({confidence:.0%})",
            raw_value   = confidence,
            metadata    = metadata,
        )
        return [item]

    # ── EIE Evidence ──────────────────────────────────────────────────────────

    def _collect_eie(self, fi: FusionInput) -> List[EvidenceItem]:
        if not fi.eie_reports:
            return [self._absent_item(
                SourceType.FUNDAMENTAL_EIE, "EIE — no active reports"
            )]

        # Filter to reports with sufficient remaining influence
        active = [
            r for r in fi.eie_reports
            if getattr(r, "remaining_influence", 0.0) >= dfe_config.DFE_EIE_MIN_REMAINING_INFLUENCE
        ]

        if not active:
            return [self._absent_item(
                SourceType.FUNDAMENTAL_EIE, "EIE — all events below influence threshold"
            )]

        # Build one aggregated evidence item from the active reports.
        # Weighted aggregation: higher impact_score and remaining_influence = more weight.
        bullish_weight = 0.0
        bearish_weight = 0.0
        neutral_weight = 0.0
        total_weight   = 0.0
        total_conf_num = 0.0
        top_report     = active[0]
        top_importance = 0.0
        timestamp      = fi.current_time

        for r in active:
            impact   = getattr(r, "impact_score", 50.0)
            remain   = getattr(r, "remaining_influence", 50.0)
            dir_conf = getattr(r, "direction_confidence", 0.5)
            raw_dir  = getattr(r, "economic_direction", None)
            dir_str  = (raw_dir.value if hasattr(raw_dir, "value") else str(raw_dir)).upper()
            w = (impact / 100.0) * (remain / 100.0)

            if dir_str == "BULLISH":
                bullish_weight += w
            elif dir_str == "BEARISH":
                bearish_weight += w
            else:
                neutral_weight += w

            total_weight   += w
            total_conf_num += dir_conf * w

            # Track highest-importance report for metadata
            imp_val = _IMPORTANCE_MAP.get(
                getattr(r, "importance", "MEDIUM").value
                if hasattr(getattr(r, "importance", None), "value")
                else str(getattr(r, "importance", "MEDIUM")),
                0.7,
            )
            if imp_val > top_importance:
                top_importance = imp_val
                top_report = r

            r_ts = getattr(r, "generated_at", None) or getattr(r, "last_updated", None)
            if r_ts is not None:
                timestamp = r_ts

        # Resolve aggregated direction
        if total_weight == 0.0:
            direction = EvidenceDirection.NEUTRAL
            agg_conf  = 0.5
        elif bullish_weight > bearish_weight and bullish_weight > neutral_weight:
            direction = EvidenceDirection.BULLISH
            agg_conf  = total_conf_num / total_weight
        elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
            direction = EvidenceDirection.BEARISH
            agg_conf  = total_conf_num / total_weight
        else:
            direction = EvidenceDirection.NEUTRAL
            agg_conf  = total_conf_num / total_weight if total_weight > 0 else 0.5

        raw_dir_top = getattr(top_report, "economic_direction", "NEUTRAL")
        raw_dir_str = (
            raw_dir_top.value if hasattr(raw_dir_top, "value") else str(raw_dir_top)
        )

        importance = min(1.0, top_importance)

        item = EvidenceItem(
            source      = SourceType.FUNDAMENTAL_EIE,
            direction   = direction,
            confidence  = round(max(0.0, min(1.0, agg_conf)), 3),
            reliability = dfe_config.DFE_RELIABILITY_EIE,
            importance  = importance,
            timestamp   = timestamp,
            label       = (
                f"EIE {direction.value} ({len(active)} active events, "
                f"top: {getattr(top_report, 'event_title', 'unknown')})"
            ),
            raw_value   = round(agg_conf, 3),
            metadata    = {
                "active_event_count":  len(active),
                "top_event_title":     getattr(top_report, "event_title", ""),
                "top_currency":        getattr(top_report, "currency", ""),
                "top_direction":       raw_dir_str,
                "execution_risk":      fi.eie_execution_risk,
                "execution_readiness": fi.eie_execution_readiness,
                "bullish_weight":      round(bullish_weight, 3),
                "bearish_weight":      round(bearish_weight, 3),
            },
        )
        return [item]

    # ── MIA Evidence ──────────────────────────────────────────────────────────

    def _collect_mia(self, fi: FusionInput) -> List[EvidenceItem]:
        if fi.mia_output is None:
            return [self._absent_item(
                SourceType.AI_INTELLIGENCE, "MIA — no analysis available"
            )]

        raw_bias = fi.mia_bias or "UNCERTAIN"
        direction = _MIA_BIAS_MAP.get(raw_bias.upper(), EvidenceDirection.UNCERTAIN)
        confidence = max(0.0, min(1.0, fi.mia_confidence or 0.0))

        # Don't use fallback MIA results — they have is_fallback=True
        is_fallback = getattr(fi.mia_output, "is_fallback", False)
        if is_fallback:
            return [self._absent_item(
                SourceType.AI_INTELLIGENCE, "MIA — fallback response (AI unavailable)"
            )]

        raw_importance = getattr(fi.mia_output, "importance", None)
        imp_str = (
            raw_importance.value
            if hasattr(raw_importance, "value")
            else str(raw_importance or "MEDIUM")
        )
        importance = _IMPORTANCE_MAP.get(imp_str.upper(), 0.7)

        execution_warning = getattr(fi.mia_output, "execution_warning", None)
        market_summary    = getattr(fi.mia_output, "market_summary", "")

        item = EvidenceItem(
            source      = SourceType.AI_INTELLIGENCE,
            direction   = direction,
            confidence  = confidence,
            reliability = dfe_config.DFE_RELIABILITY_MIA,
            importance  = importance,
            timestamp   = getattr(fi.mia_output, "timestamp", fi.current_time),
            label       = f"MIA {raw_bias} ({confidence:.0%})",
            raw_value   = confidence,
            metadata    = {
                "risk_level":           fi.mia_risk_level,
                "execution_warning":    execution_warning,
                "market_summary":       market_summary,
                "supports_existing":    getattr(fi.mia_output, "supports_existing_bias", None),
                "contradicts_existing": getattr(fi.mia_output, "contradicts_existing_bias", None),
                "affected_currencies":  getattr(fi.mia_output, "affected_currencies", []),
            },
        )
        return [item]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _absent_item(source: SourceType, label: str) -> EvidenceItem:
        return EvidenceItem(
            source      = source,
            direction   = EvidenceDirection.ABSENT,
            confidence  = 0.0,
            reliability = 0.0,
            importance  = 0.0,
            timestamp   = datetime.now(timezone.utc),
            label       = label,
        )
