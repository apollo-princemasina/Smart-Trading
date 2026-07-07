"""
Infers EventCategory, is_speech, and country from Forex Factory raw event fields.

All keyword matching is case-insensitive and operates on the event title.
Order matters inside each group — more specific patterns come first.
"""
from market_intel.models.enums import EventCategory

# Maps FF currency codes to ISO 3166-1 alpha-2 country codes.
# FF uses currency as a proxy for country; "EU" is non-standard but widely used.
CURRENCY_TO_COUNTRY: dict[str, str] = {
    "USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP",
    "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ",
    "CNY": "CN", "HKD": "HK", "SGD": "SG", "NOK": "NO",
    "SEK": "SE", "DKK": "DK", "MXN": "MX", "BRL": "BR",
    "INR": "IN", "KRW": "KR", "ZAR": "ZA", "TRY": "TR",
}

_SPEECH_KEYWORDS = frozenset({
    "speaks", "speech", "testimony", "presser", "press conference",
    "statement", "remarks", "hearing", "vote",
})

# Ordered list of (keywords, category) — first match wins.
_CATEGORY_RULES: list[tuple[frozenset[str], EventCategory]] = [
    (frozenset({"speaks", "speech", "testimony", "presser", "press conference",
                "statement", "remarks", "hearing"}),           EventCategory.SPEECH),
    (frozenset({"rate decision", "interest rate", "fomc", "boe", "ecb", "boj",
                "rba", "rbnz", "fed", "monetary policy", "basis points"}),
                                                               EventCategory.CENTRAL_BANK),
    (frozenset({"non-farm", "nonfarm", "employment change", "unemployment",
                "payroll", "jobless", "labor", "labour", "jobs report",
                "claimant", "participation rate", "average hourly"}),
                                                               EventCategory.EMPLOYMENT),
    (frozenset({"cpi", "ppi", "inflation", "price index", "deflator",
                "core prices", "pcе", "pce"}),                EventCategory.INFLATION),
    (frozenset({"gdp", "gross domestic"}),                    EventCategory.GDP),
    (frozenset({"trade balance", "current account", "exports", "imports",
                "trade deficit", "trade surplus"}),            EventCategory.TRADE),
    (frozenset({"retail sales", "consumer spending"}),         EventCategory.RETAIL),
    (frozenset({"housing starts", "building permits", "house prices",
                "hpi", "home sales", "existing home", "new home"}),
                                                               EventCategory.HOUSING),
    (frozenset({"pmi", "ism", "manufacturing", "industrial production",
                "factory orders", "durable goods"}),           EventCategory.MANUFACTURING),
    (frozenset({"consumer confidence", "consumer sentiment", "cci", "zew",
                "ifo", "sentix", "business climate"}),         EventCategory.SENTIMENT),
]


def infer_is_speech(title: str) -> bool:
    lower = title.lower()
    return any(kw in lower for kw in _SPEECH_KEYWORDS)


def infer_category(title: str) -> EventCategory:
    lower = title.lower()
    for keywords, category in _CATEGORY_RULES:
        if any(kw in lower for kw in keywords):
            return category
    return EventCategory.OTHER


def currency_to_country(currency: str) -> str:
    return CURRENCY_TO_COUNTRY.get(currency.upper(), currency.upper())
