"""Unit tests for secrag/decompose.py's payload parsing and fail-open behavior.

No live `claude` CLI calls — subprocess.run is mocked so these stay fast/isolated,
matching the existing test style (tests/test_calculator.py's _parse_payload tests).
"""
import json
import subprocess
from unittest.mock import patch

import pytest

from secrag.decompose import DecomposeError, _parse_payload, decompose


def _fake_result(stdout_result: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps({"result": stdout_result}))


def test_parse_payload_valid_single():
    parsed = _parse_payload('{"sub_queries": ["total revenue for fiscal year 2019"]}')
    assert parsed == {"sub_queries": ["total revenue for fiscal year 2019"]}


def test_parse_payload_valid_multi():
    parsed = _parse_payload('{"sub_queries": ["revenue for FY2019", "PP&E net value for FY2019"]}')
    assert len(parsed["sub_queries"]) == 2


def test_parse_payload_strips_json_fence():
    raw = '```json\n{"sub_queries": ["revenue for FY2019"]}\n```'
    assert _parse_payload(raw) == {"sub_queries": ["revenue for FY2019"]}


def test_parse_payload_invalid_json_raises():
    with pytest.raises(DecomposeError):
        _parse_payload("not json at all")


def test_parse_payload_missing_key_raises():
    with pytest.raises(DecomposeError):
        _parse_payload('{"other_key": ["x"]}')


def test_parse_payload_empty_list_raises():
    with pytest.raises(DecomposeError):
        _parse_payload('{"sub_queries": []}')


def test_parse_payload_non_string_entries_raises():
    with pytest.raises(DecomposeError):
        _parse_payload('{"sub_queries": [123]}')


def test_decompose_returns_sub_queries_on_success():
    with patch("subprocess.run", return_value=_fake_result('{"sub_queries": ["a", "b"]}')):
        assert decompose("some question") == ["a", "b"]


def test_decompose_falls_back_on_cli_failure():
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, [])):
        assert decompose("some question") == ["some question"]


def test_decompose_falls_back_on_malformed_json():
    with patch("subprocess.run", return_value=_fake_result("not json")):
        assert decompose("some question") == ["some question"]


def test_decompose_falls_back_on_missing_result_key():
    bad_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps({"no_result": "x"}))
    with patch("subprocess.run", return_value=bad_result):
        assert decompose("some question") == ["some question"]
