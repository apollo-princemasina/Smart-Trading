"""Tests for ResponseValidator against MarketIntelligenceOutput schema."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from market_intelligence_ai.response_validator.validator import ResponseValidator, ValidationFailure
from market_intelligence_ai.schema.market_intelligence import MarketIntelligenceOutput

_VALID_JSON = {
    "market_bias":               "BULLISH",
    "affected_currencies":       ["USD"],
    "importance":                "HIGH",
    "confidence":                0.82,
    "expected_duration":         "SHORT_TERM",
    "supports_existing_bias":    True,
    "contradicts_existing_bias": False,
    "risk_level":                "LOW",
    "execution_warning":         None,
    "market_summary":            "Strong US data drives USD demand.",
    "timestamp":                 "2026-07-01T13:30:00+00:00",
    "latency_ms":                320.5,
}


def test_validates_correct_json():
    raw = json.dumps(_VALID_JSON)
    result = ResponseValidator.validate(raw, MarketIntelligenceOutput)
    assert result.market_bias.value == "BULLISH"
    assert result.confidence == 0.82
    assert result.risk_level.value == "LOW"
    assert result.is_fallback is False


def test_strips_markdown_fences():
    raw = "```json\n" + json.dumps(_VALID_JSON) + "\n```"
    result = ResponseValidator.validate(raw, MarketIntelligenceOutput)
    assert result.market_bias.value == "BULLISH"


def test_strips_leading_prose():
    raw = "Here is my analysis:\n\n" + json.dumps(_VALID_JSON)
    result = ResponseValidator.validate(raw, MarketIntelligenceOutput)
    assert result.confidence == 0.82


def test_raises_on_empty_content():
    with pytest.raises(ValidationFailure, match="empty content"):
        ResponseValidator.validate("", MarketIntelligenceOutput)


def test_raises_on_no_json():
    with pytest.raises(ValidationFailure, match="No JSON object"):
        ResponseValidator.validate("This is plain text with no JSON.", MarketIntelligenceOutput)


def test_raises_on_invalid_market_bias_enum():
    bad = dict(_VALID_JSON, market_bias="VERY_BULLISH")
    with pytest.raises(ValidationFailure, match="Schema validation failed"):
        ResponseValidator.validate(json.dumps(bad), MarketIntelligenceOutput)


def test_raises_on_invalid_risk_level_enum():
    bad = dict(_VALID_JSON, risk_level="EXTREME")  # not a valid RiskLevel
    with pytest.raises(ValidationFailure, match="Schema validation failed"):
        ResponseValidator.validate(json.dumps(bad), MarketIntelligenceOutput)


def test_raises_on_confidence_out_of_range():
    bad = dict(_VALID_JSON, confidence=1.5)
    with pytest.raises(ValidationFailure, match="Schema validation failed"):
        ResponseValidator.validate(json.dumps(bad), MarketIntelligenceOutput)


def test_raises_on_missing_required_field_market_summary():
    bad = {k: v for k, v in _VALID_JSON.items() if k != "market_summary"}
    with pytest.raises(ValidationFailure, match="Schema validation failed"):
        ResponseValidator.validate(json.dumps(bad), MarketIntelligenceOutput)


def test_raises_on_missing_risk_level():
    bad = {k: v for k, v in _VALID_JSON.items() if k != "risk_level"}
    with pytest.raises(ValidationFailure, match="Schema validation failed"):
        ResponseValidator.validate(json.dumps(bad), MarketIntelligenceOutput)


def test_raises_on_missing_expected_duration():
    bad = {k: v for k, v in _VALID_JSON.items() if k != "expected_duration"}
    with pytest.raises(ValidationFailure, match="Schema validation failed"):
        ResponseValidator.validate(json.dumps(bad), MarketIntelligenceOutput)


def test_build_repair_prompt_contains_error():
    prompt = ResponseValidator.build_repair_prompt("original prompt", "confidence must be <= 1.0")
    assert "confidence must be <= 1.0" in prompt
    assert "RETRY" in prompt.upper()


def test_make_fallback_sets_is_fallback():
    result = ResponseValidator.make_fallback(MarketIntelligenceOutput, {
        "market_bias":               "UNCERTAIN",
        "affected_currencies":       ["USD"],
        "importance":                "LOW",
        "confidence":                0.0,
        "expected_duration":         "SHORT_TERM",
        "supports_existing_bias":    False,
        "contradicts_existing_bias": False,
        "risk_level":                "MEDIUM",
        "execution_warning":         "unavailable",
        "market_summary":            "Fallback.",
        "timestamp":                 datetime.now(timezone.utc),
        "latency_ms":                0.0,
        "is_fallback":               True,
    })
    assert result.is_fallback is True
    assert result.market_bias.value == "UNCERTAIN"
    assert result.risk_level.value == "MEDIUM"
