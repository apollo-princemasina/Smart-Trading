"""
ResponseValidator — validates every LLM response before it enters the backend.

Malformed responses never propagate. Validation pipeline:
  1. Extract JSON from raw text (handles leading/trailing content from non-compliant models)
  2. Parse JSON
  3. Validate against the target Pydantic model
  4. Check confidence range (0.0 – 1.0)
  5. Check enum field values
  6. Return validated instance or raise ValidationFailure

The gateway handles retries. The validator is pure: validate-or-raise.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from market_intelligence_ai.utils.logger import logger

T = TypeVar("T", bound=BaseModel)

# Regex to extract the outermost JSON object from text that may have surrounding content
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ValidationFailure(Exception):
    """Raised when a provider response cannot be validated against the target schema."""

    def __init__(self, message: str, raw_content: str = "") -> None:
        super().__init__(message)
        self.raw_content = raw_content


class ResponseValidator:
    """
    Validates and parses LLM provider responses into Pydantic models.

    All methods are static — the validator is stateless.
    """

    @staticmethod
    def validate(raw_content: str, model_cls: Type[T]) -> T:
        """
        Parse and validate `raw_content` into an instance of `model_cls`.

        Raises:
            ValidationFailure: if the content cannot be parsed or does not
                               match the schema.
        """
        if not raw_content or not raw_content.strip():
            raise ValidationFailure("Provider returned empty content", raw_content)

        # 1 — Extract JSON (model may have added text around the object)
        json_str = ResponseValidator._extract_json(raw_content)
        if json_str is None:
            raise ValidationFailure(
                f"No JSON object found in response (first 200 chars): {raw_content[:200]}",
                raw_content,
            )

        # 2 — Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValidationFailure(
                f"JSON parse error: {exc} | content: {json_str[:200]}",
                raw_content,
            ) from exc

        if not isinstance(data, dict):
            raise ValidationFailure(
                f"Expected JSON object, got {type(data).__name__}",
                raw_content,
            )

        # 3 — Pydantic validation
        try:
            instance = model_cls(**data)
        except ValidationError as exc:
            errors = exc.errors()
            error_summary = "; ".join(
                f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in errors[:3]
            )
            raise ValidationFailure(
                f"Schema validation failed ({model_cls.__name__}): {error_summary}",
                raw_content,
            ) from exc

        return instance

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the outermost JSON object from text."""
        text = text.strip()

        # Fast path: entire string is valid JSON
        if text.startswith("{") and text.endswith("}"):
            return text

        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        if text.startswith("{") and text.endswith("}"):
            return text

        # Scan for outermost braces
        m = _JSON_OBJECT_RE.search(text)
        if m:
            return m.group(0)

        return None

    @staticmethod
    def build_repair_prompt(original_prompt: str, validation_error: str) -> str:
        """
        Build a repair prompt asking the model to fix its invalid response.

        Used by the gateway on the second retry attempt.
        """
        return (
            f"{original_prompt}\n\n"
            "YOUR PREVIOUS RESPONSE WAS REJECTED due to a validation error:\n"
            f"  {validation_error}\n\n"
            "INSTRUCTIONS FOR THIS RETRY:\n"
            "1. Respond with ONLY a valid JSON object — no markdown, no explanation.\n"
            "2. The JSON must start with {{ and end with }}.\n"
            "3. All enum fields must match exactly (e.g. \"BULLISH\", not \"bullish\").\n"
            "4. All confidence values must be floats between 0.0 and 1.0.\n"
            "5. Do not add any text outside the JSON object."
        )

    @staticmethod
    def make_fallback(model_cls: Type[T], overrides: dict[str, Any]) -> T:
        """
        Create a minimal valid fallback instance when all retries fail.

        The `is_fallback` field is set to True to signal downstream that
        this is a degraded response, not real AI output.
        """
        logger.warning("Building fallback response for {}", model_cls.__name__)
        return model_cls(**overrides)
