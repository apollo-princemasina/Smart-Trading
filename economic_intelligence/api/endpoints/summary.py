"""GET /intelligence/economic-summary — per-currency economic direction summary."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter

from economic_intelligence.intelligence_cache.cache import intelligence_cache
from economic_intelligence.direction_engine.models import EconomicDirection
from economic_intelligence.api.schemas import (
    EconomicSummaryResponse,
    CurrencyDirectionOut,
)

router = APIRouter()


def _dominant_direction(reports) -> EconomicDirection:
    counts: dict[EconomicDirection, int] = defaultdict(int)
    for r in reports:
        counts[r.economic_direction] += 1

    # Ignore UNCERTAIN in dominant direction calculation
    filtered = {k: v for k, v in counts.items() if k != EconomicDirection.UNCERTAIN}
    if not filtered:
        return EconomicDirection.UNCERTAIN

    return max(filtered, key=lambda k: filtered[k])


@router.get("/economic-summary", response_model=EconomicSummaryResponse)
async def get_economic_summary():
    """
    Economic direction summary grouped by currency.

    Aggregates all active events to give a per-currency directional bias.
    Useful for quickly assessing which currencies have strong economic signals.
    """
    all_reports = await intelligence_cache.get_all()

    if not all_reports:
        return EconomicSummaryResponse(
            currencies=[],
            generated_at=datetime.now(timezone.utc),
            total_reports=0,
        )

    # Group by currency
    by_currency: dict[str, list] = defaultdict(list)
    for r in all_reports:
        by_currency[r.currency].append(r)

    summaries: list[CurrencyDirectionOut] = []
    for currency, reports in sorted(by_currency.items()):
        bullish  = sum(1 for r in reports if r.economic_direction == EconomicDirection.BULLISH)
        bearish  = sum(1 for r in reports if r.economic_direction == EconomicDirection.BEARISH)
        neutral  = sum(1 for r in reports if r.economic_direction == EconomicDirection.NEUTRAL)

        avg_conf    = sum(r.direction_confidence for r in reports) / len(reports)
        avg_impact  = sum(r.impact_score         for r in reports) / len(reports)
        avg_remain  = sum(r.remaining_influence  for r in reports) / len(reports)

        summaries.append(CurrencyDirectionOut(
            currency=currency,
            dominant_direction=_dominant_direction(reports),
            avg_confidence=round(avg_conf, 3),
            avg_impact_score=round(avg_impact, 1),
            avg_remaining_influence=round(avg_remain, 1),
            active_event_count=len(reports),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        ))

    return EconomicSummaryResponse(
        currencies=summaries,
        generated_at=datetime.now(timezone.utc),
        total_reports=len(all_reports),
    )
