"""
Impact Engine weight tables.

All values are tuned against observed market reactions to typical economic releases.
"""
from __future__ import annotations

from market_intel.models.enums import ImpactLevel
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.surprise_engine.models import SurpriseClass

# ── Base scores by FF impact level ────────────────────────────────────────────
# These represent the pre-release importance of the event.
BASE_IMPACT_SCORES: dict[ImpactLevel, float] = {
    ImpactLevel.HIGH:         90.0,
    ImpactLevel.MEDIUM:       50.0,
    ImpactLevel.LOW:          20.0,
    ImpactLevel.HOLIDAY:       5.0,
    ImpactLevel.NON_ECONOMIC:  2.0,
}

# ── Type multipliers ───────────────────────────────────────────────────────────
# Scale the base score based on the economic significance of the event type.
# Range: 0.0 – 1.0 (values > 1.0 intentionally excluded to keep max at ~90)
EVENT_TYPE_MULTIPLIERS: dict[EventType, float] = {
    EventType.INTEREST_RATE:          1.00,
    EventType.EMPLOYMENT:             0.95,
    EventType.WAGES:                  0.88,
    EventType.INFLATION:              0.90,
    EventType.UNEMPLOYMENT:           0.85,
    EventType.JOBLESS_CLAIMS:         0.70,
    EventType.GDP:                    0.85,
    EventType.CENTRAL_BANK_SPEECH:    0.75,
    EventType.PMI:                    0.72,
    EventType.RETAIL_SALES:           0.70,
    EventType.CONSUMER_CONFIDENCE:    0.65,
    EventType.MANUFACTURING:          0.65,
    EventType.INDUSTRIAL:             0.60,
    EventType.TRADE_BALANCE:          0.62,
    EventType.HOUSING:                0.58,
    EventType.OIL_INVENTORY:          0.55,
    EventType.POLITICAL:              0.40,
    EventType.HOLIDAY:                0.05,
    EventType.UNKNOWN:                0.50,
}

_DEFAULT_TYPE_MULTIPLIER = 0.50

# ── Surprise amplifiers ────────────────────────────────────────────────────────
# Applied AFTER the event is released — amplifies or dampens the impact score
# based on how surprising the actual reading was.
SURPRISE_AMPLIFIERS: dict[SurpriseClass, float] = {
    SurpriseClass.NONE:    1.00,  # in-line: no change
    SurpriseClass.SMALL:   1.08,  # slight beat/miss: +8%
    SurpriseClass.MEDIUM:  1.20,  # notable: +20%
    SurpriseClass.LARGE:   1.40,  # significant: +40%
    SurpriseClass.EXTREME: 1.65,  # shock: +65%
}
