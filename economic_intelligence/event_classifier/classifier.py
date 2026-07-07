"""
EventClassifier — maps a MFIPEvent to a granular EventType.

Uses keyword rules (fastest path) with a fallback to EventCategory mapping.
All logic is deterministic and stateless.
"""
from __future__ import annotations

from market_intel.models.enums import EventCategory, ImpactLevel
from market_intel.models.event import MFIPEvent
from economic_intelligence.event_classifier.event_types import EventType
from economic_intelligence.event_classifier.rules import COMPILED_RULES

# Fallback: map MFIPEvent.category → EventType when keyword rules miss
_CATEGORY_FALLBACK: dict[EventCategory, EventType] = {
    EventCategory.EMPLOYMENT:    EventType.EMPLOYMENT,
    EventCategory.INFLATION:     EventType.INFLATION,
    EventCategory.GDP:           EventType.GDP,
    EventCategory.TRADE:         EventType.TRADE_BALANCE,
    EventCategory.CENTRAL_BANK:  EventType.INTEREST_RATE,
    EventCategory.HOUSING:       EventType.HOUSING,
    EventCategory.MANUFACTURING: EventType.MANUFACTURING,
    EventCategory.RETAIL:        EventType.RETAIL_SALES,
    EventCategory.SENTIMENT:     EventType.CONSUMER_CONFIDENCE,
    EventCategory.SPEECH:        EventType.CENTRAL_BANK_SPEECH,
}


class EventClassifier:
    """
    Classifies a MFIPEvent into a granular EventType.

    Classification strategy (first match wins):
    1. HOLIDAY impact level → EventType.HOLIDAY
    2. Keyword rules against title (COMPILED_RULES)
    3. EventCategory fallback mapping
    4. EventType.UNKNOWN
    """

    @staticmethod
    def classify(event: MFIPEvent) -> EventType:
        if event.impact == ImpactLevel.HOLIDAY:
            return EventType.HOLIDAY

        title_lower = event.title.lower()

        for keywords, event_type in COMPILED_RULES:
            if any(kw in title_lower for kw in keywords):
                return event_type

        fallback = _CATEGORY_FALLBACK.get(event.category)
        if fallback is not None:
            return fallback

        return EventType.UNKNOWN

    @staticmethod
    def classify_title(title: str) -> EventType:
        """Classify by title string alone — useful for unit tests."""
        title_lower = title.lower()
        for keywords, event_type in COMPILED_RULES:
            if any(kw in title_lower for kw in keywords):
                return event_type
        return EventType.UNKNOWN
