"""
Decay profiles — defines how each event type's influence fades over time.

Decay formula (exponential half-life):
  remaining = base_score * (0.5 ** (hours_elapsed / half_life_hours))
  remaining = max(min_influence, remaining)

half_life_hours: Hours until influence halves.
  - FOMC/rate decisions persist longest (institutional memory, repricing)
  - NFP/CPI persist for 8-12 h (strong but fade by end of session)
  - PMI/confidence fade within 4 h (technical, lower volatility)

min_influence: Floor value — events never drop completely to 0 within the week.
  Represents the residual awareness traders have of the event.
"""
from __future__ import annotations

from dataclasses import dataclass
from economic_intelligence.event_classifier.event_types import EventType


@dataclass(frozen=True)
class DecayProfile:
    half_life_hours: float  # Hours for influence to halve
    min_influence:   float  # Minimum remaining influence (0–100)


DECAY_PROFILES: dict[EventType, DecayProfile] = {
    EventType.INTEREST_RATE:         DecayProfile(half_life_hours=24.0, min_influence=15.0),
    EventType.CENTRAL_BANK_SPEECH:   DecayProfile(half_life_hours=12.0, min_influence=10.0),
    EventType.EMPLOYMENT:            DecayProfile(half_life_hours=8.0,  min_influence=5.0),
    EventType.WAGES:                 DecayProfile(half_life_hours=8.0,  min_influence=5.0),
    EventType.UNEMPLOYMENT:          DecayProfile(half_life_hours=8.0,  min_influence=5.0),
    EventType.INFLATION:             DecayProfile(half_life_hours=12.0, min_influence=5.0),
    EventType.GDP:                   DecayProfile(half_life_hours=8.0,  min_influence=3.0),
    EventType.RETAIL_SALES:          DecayProfile(half_life_hours=6.0,  min_influence=3.0),
    EventType.JOBLESS_CLAIMS:        DecayProfile(half_life_hours=4.0,  min_influence=2.0),
    EventType.PMI:                   DecayProfile(half_life_hours=4.0,  min_influence=2.0),
    EventType.MANUFACTURING:         DecayProfile(half_life_hours=4.0,  min_influence=2.0),
    EventType.INDUSTRIAL:            DecayProfile(half_life_hours=4.0,  min_influence=2.0),
    EventType.CONSUMER_CONFIDENCE:   DecayProfile(half_life_hours=4.0,  min_influence=2.0),
    EventType.TRADE_BALANCE:         DecayProfile(half_life_hours=6.0,  min_influence=2.0),
    EventType.HOUSING:               DecayProfile(half_life_hours=4.0,  min_influence=1.0),
    EventType.OIL_INVENTORY:         DecayProfile(half_life_hours=3.0,  min_influence=1.0),
    EventType.POLITICAL:             DecayProfile(half_life_hours=6.0,  min_influence=2.0),
}

_DEFAULT_DECAY_PROFILE = DecayProfile(half_life_hours=4.0, min_influence=1.0)


def get_decay_profile(event_type: EventType) -> DecayProfile:
    return DECAY_PROFILES.get(event_type, _DEFAULT_DECAY_PROFILE)
