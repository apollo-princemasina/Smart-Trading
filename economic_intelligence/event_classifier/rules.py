"""
Keyword-rule registry for EventType classification.

Rules are ordered lists of (keyword_set, EventType) pairs.
The classifier iterates them and returns the FIRST match.

Keyword matching is case-insensitive against the full event title.
"""
from __future__ import annotations

from economic_intelligence.event_classifier.event_types import EventType

# Each entry: (frozenset of keywords, EventType)
# At least one keyword from the set must appear in the lowercased title.
_CLASSIFICATION_RULES: list[tuple[frozenset[str], EventType]] = [
    # ── Central Bank speeches (before INTEREST_RATE so "speaks" doesn't match rate)
    (frozenset({
        "speaks", "speech", "testimony", "testifies", "presser",
        "press conference", "news conference", "forum", "panel",
        "minutes", "fed chair", "boe governor", "ecb president",
        "rba governor", "boc governor", "snb chair",
    }), EventType.CENTRAL_BANK_SPEECH),

    # ── Interest Rate decisions
    (frozenset({
        "interest rate", "rate decision", "monetary policy",
        "bank rate", "cash rate", "overnight rate", "benchmark rate",
        "fomc statement", "fomc", "fed funds", "boe decision",
        "ecb decision", "rba decision", "boc decision", "snb decision",
        "rate vote", "rate statement",
    }), EventType.INTEREST_RATE),

    # ── Employment / Payrolls
    (frozenset({
        "non-farm", "nonfarm", "employment change", "payrolls",
        "jobs report", "jobs added", "adp employment",
        "private payrolls", "public sector employment",
        "employment level", "claimant count change",
    }), EventType.EMPLOYMENT),

    # ── Unemployment Rate
    (frozenset({
        "unemployment rate", "unemployment",
        "participation rate", "labor force",
    }), EventType.UNEMPLOYMENT),

    # ── Jobless Claims
    (frozenset({
        "jobless claims", "initial claims", "continuing claims",
        "unemployment claims", "weekly claims",
    }), EventType.JOBLESS_CLAIMS),

    # ── Wages
    (frozenset({
        "average hourly earnings", "hourly earnings",
        "wage growth", "wage price", "wage inflation",
        "compensation", "labor costs", "unit labor",
    }), EventType.WAGES),

    # ── Inflation (CPI / PCE / PPI)
    (frozenset({
        "cpi", "consumer price", "inflation", "pce",
        "personal consumption expenditure", "ppi", "producer price",
        "rpi", "retail price", "core inflation", "core cpi",
        "headline cpi", "trimmed mean", "deflator",
    }), EventType.INFLATION),

    # ── GDP
    (frozenset({
        "gdp", "gross domestic product",
        "economic growth", "output growth",
    }), EventType.GDP),

    # ── PMI
    (frozenset({
        "pmi", "purchasing managers", "composite pmi",
        "services pmi", "manufacturing pmi",
    }), EventType.PMI),

    # ── Manufacturing
    (frozenset({
        "manufacturing", "ism manufacturing", "factory orders",
        "industrial orders", "durable goods",
    }), EventType.MANUFACTURING),

    # ── Industrial Production
    (frozenset({
        "industrial production", "capacity utilization",
        "factory output", "production output",
    }), EventType.INDUSTRIAL),

    # ── Retail Sales
    (frozenset({
        "retail sales", "core retail", "consumer spending",
        "household spending",
    }), EventType.RETAIL_SALES),

    # ── Consumer Confidence
    (frozenset({
        "consumer confidence", "consumer sentiment",
        "consumer expectations", "university of michigan",
        "umich", "conference board", "zew",
    }), EventType.CONSUMER_CONFIDENCE),

    # ── Trade Balance
    (frozenset({
        "trade balance", "current account", "trade deficit",
        "trade surplus", "goods trade", "merchandise trade",
        "exports", "imports",
    }), EventType.TRADE_BALANCE),

    # ── Housing
    (frozenset({
        "housing starts", "housing permits", "building permits",
        "existing home sales", "new home sales", "pending home",
        "housing", "construction spending", "case-shiller",
    }), EventType.HOUSING),

    # ── Oil Inventory
    (frozenset({
        "crude oil", "oil inventories", "eia", "petroleum",
        "distillate", "gasoline", "oil stocks",
    }), EventType.OIL_INVENTORY),

    # ── Political
    (frozenset({
        "election", "referendum", "vote", "budget", "fiscal",
        "government shutdown", "debt ceiling",
    }), EventType.POLITICAL),
]

# Pre-processed for faster classification: list of (lowered_keywords_frozenset, EventType)
COMPILED_RULES: list[tuple[frozenset[str], EventType]] = _CLASSIFICATION_RULES
