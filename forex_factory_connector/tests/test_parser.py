import json
import pytest
from ..parser.calendar_parser import parse_calendar_json


def test_parse_valid_json():
    raw = json.dumps([{"title": "Test", "country": "USD", "date": "Jul 04, 2026",
                       "time": "8:30am", "impact": "High", "forecast": "", "previous": "", "actual": ""}])
    result = parse_calendar_json(raw.encode())
    assert isinstance(result, list)
    assert len(result) == 1


def test_parse_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_calendar_json(b"not json{{{")


def test_parse_wrong_root_type():
    with pytest.raises(ValueError, match="Expected JSON array"):
        parse_calendar_json(b'{"key": "value"}')
