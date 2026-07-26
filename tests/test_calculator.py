"""Unit tests for secrag/calculator.py's sandbox (no network/CLI calls)."""
import pytest

from secrag.calculator import CalculatorError, _parse_payload, run_sandboxed


def test_basic_decimal_arithmetic():
    out = run_sandboxed("from decimal import Decimal; print(Decimal('1577') / Decimal('1'))")
    assert out == "1577"


def test_disallowed_import_raises():
    with pytest.raises(CalculatorError):
        run_sandboxed("import os; print(os.getcwd())")


def test_timeout_raises():
    with pytest.raises(CalculatorError):
        run_sandboxed("while True: pass")


def test_oversized_output_raises():
    with pytest.raises(CalculatorError):
        run_sandboxed("while True: print('x' * 10000)")


def test_real_import_escape_is_blocked():
    # Regression: the first cut of this sandbox concatenated code into the preamble's own
    # globals, leaving its `_real_import` name reachable and bypassing the allowlist.
    with pytest.raises(CalculatorError):
        run_sandboxed("print(_real_import('os').getcwd())")


def test_open_is_blocked():
    # Regression: builtins were previously untouched, so `open` could read arbitrary files.
    with pytest.raises(CalculatorError):
        run_sandboxed("print(open('/etc/hosts').read())")


def test_eval_and_exec_are_blocked():
    with pytest.raises(CalculatorError):
        run_sandboxed("eval('1+1')")
    with pytest.raises(CalculatorError):
        run_sandboxed("exec('x = 1')")


def test_parse_payload_valid():
    parsed = _parse_payload('{"code": "print(1)", "explanation": "one"}')
    assert parsed == {"code": "print(1)", "explanation": "one"}


def test_parse_payload_strips_json_fence():
    raw = '```json\n{"code": "print(1)", "explanation": "one"}\n```'
    parsed = _parse_payload(raw)
    assert parsed == {"code": "print(1)", "explanation": "one"}


def test_parse_payload_invalid_json_raises_calculator_error():
    with pytest.raises(CalculatorError):
        _parse_payload("not json at all")


def test_parse_payload_missing_keys_raises_calculator_error():
    with pytest.raises(CalculatorError):
        _parse_payload('{"code": "print(1)"}')
