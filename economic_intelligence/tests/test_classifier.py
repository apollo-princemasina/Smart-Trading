"""Unit tests for the EventClassifier."""
from __future__ import annotations

import pytest

from economic_intelligence.event_classifier.classifier import EventClassifier
from economic_intelligence.event_classifier.event_types import EventType
from market_intel.models.enums import ImpactLevel

from economic_intelligence.tests.conftest import make_event


@pytest.mark.parametrize("title,expected", [
    ("US Non-Farm Employment Change",           EventType.EMPLOYMENT),
    ("ADP Non-Farm Employment Change",          EventType.EMPLOYMENT),
    ("German Unemployment Rate",                EventType.UNEMPLOYMENT),
    ("US Initial Jobless Claims",               EventType.JOBLESS_CLAIMS),
    ("US Continuing Jobless Claims",            EventType.JOBLESS_CLAIMS),
    ("Average Hourly Earnings m/m",             EventType.WAGES),
    ("US CPI m/m",                              EventType.INFLATION),
    ("German CPI y/y",                          EventType.INFLATION),
    ("US Core PCE Price Index m/m",             EventType.INFLATION),
    ("US PPI m/m",                              EventType.INFLATION),
    ("US GDP q/q",                              EventType.GDP),
    ("UK Flash GDP",                            EventType.GDP),
    ("US ISM Manufacturing PMI",                EventType.PMI),
    ("UK Services PMI",                         EventType.PMI),
    ("UK Manufacturing PMI",                    EventType.PMI),
    ("US ISM Manufacturing",                    EventType.MANUFACTURING),
    ("US Core Durable Goods Orders m/m",        EventType.MANUFACTURING),
    ("US Industrial Production m/m",            EventType.INDUSTRIAL),
    ("US Retail Sales m/m",                     EventType.RETAIL_SALES),
    ("CB Consumer Confidence",                  EventType.CONSUMER_CONFIDENCE),
    ("UoM Consumer Sentiment",                  EventType.CONSUMER_CONFIDENCE),
    ("US Trade Balance",                        EventType.TRADE_BALANCE),
    ("US Current Account",                      EventType.TRADE_BALANCE),
    ("US Housing Starts",                       EventType.HOUSING),
    ("US Building Permits",                     EventType.HOUSING),
    ("EIA Crude Oil Inventories",               EventType.OIL_INVENTORY),
    ("Fed Chair Powell Speaks",                 EventType.CENTRAL_BANK_SPEECH),
    ("ECB President Lagarde Speaks",            EventType.CENTRAL_BANK_SPEECH),
    ("BOE Governor Bailey Testimony",           EventType.CENTRAL_BANK_SPEECH),
    ("FOMC Meeting Minutes",                    EventType.CENTRAL_BANK_SPEECH),
    ("FOMC Statement",                          EventType.INTEREST_RATE),
    ("BOE Interest Rate Decision",              EventType.INTEREST_RATE),
    ("ECB Monetary Policy Decision",            EventType.INTEREST_RATE),
])
def test_classify_by_title(title: str, expected: EventType):
    result = EventClassifier.classify_title(title)
    assert result == expected, f"Expected {expected} for '{title}', got {result}"


def test_classify_holiday_event():
    event = make_event("Bank Holiday", impact=ImpactLevel.HOLIDAY)
    event_type = EventClassifier.classify(event)
    assert event_type == EventType.HOLIDAY


def test_classify_speech_takes_precedence_over_other_keywords():
    """Speech classification must win even if the title also contains employment keywords."""
    result = EventClassifier.classify_title("Fed Chair Powell Speaks — Employment Outlook")
    assert result == EventType.CENTRAL_BANK_SPEECH


def test_classify_unknown_title():
    result = EventClassifier.classify_title("Some Random Economic Indicator XYZ")
    assert result == EventType.UNKNOWN


def test_classify_case_insensitive():
    assert EventClassifier.classify_title("US CPI M/M") == EventType.INFLATION
    assert EventClassifier.classify_title("us cpi m/m") == EventType.INFLATION
    assert EventClassifier.classify_title("US Cpi M/M") == EventType.INFLATION
