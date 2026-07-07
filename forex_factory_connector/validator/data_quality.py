from market_intel.models.event import MFIPEvent
from market_intel.models.enums import ImpactLevel
from ..utils.logger import logger


def check_quality(events: list[MFIPEvent]) -> None:
    if not events:
        logger.warning("Quality check: zero MFIPEvents returned from FF CDN")
        return

    high_impact = [e for e in events if e.is_high_impact]
    currencies  = {e.currency for e in events}
    speeches    = [e for e in events if e.is_speech]

    logger.debug(
        f"Quality: {len(events)} events | "
        f"{len(high_impact)} HIGH-impact | "
        f"{len(speeches)} speeches | "
        f"currencies={sorted(currencies)}"
    )

    if "USD" not in currencies and "EUR" not in currencies:
        logger.warning(
            "Quality alert: neither USD nor EUR in FF payload — "
            "possible empty or malformed response"
        )
