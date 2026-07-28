"""Unit tests for secrag/claude_cli.py's shared invocation + metrics accumulator.

No live `claude` CLI calls — subprocess.run is mocked.
"""
import json
import subprocess
from unittest.mock import patch

from secrag.claude_cli import call_claude, get_recorded_metrics, reset_recorded_metrics


def _fake_result(result_text: str, **usage_overrides) -> subprocess.CompletedProcess:
    payload = {
        "result": result_text,
        "total_cost_usd": 0.01,
        "duration_ms": 1000,
        "duration_api_ms": 900,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 0,
            **usage_overrides,
        },
    }
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload))


def test_call_claude_returns_result_text():
    with patch("subprocess.run", return_value=_fake_result("hello")):
        reset_recorded_metrics()
        assert call_claude("prompt", "system", "sonnet") == "hello"


def test_call_claude_records_metrics():
    with patch("subprocess.run", return_value=_fake_result("hello")):
        reset_recorded_metrics()
        call_claude("prompt", "system", "sonnet")
        metrics = get_recorded_metrics()
        assert len(metrics) == 1
        assert metrics[0].input_tokens == 10
        assert metrics[0].output_tokens == 5
        assert metrics[0].cost_usd == 0.01
        assert metrics[0].duration_ms == 1000


def test_metrics_accumulate_across_multiple_calls():
    with patch("subprocess.run", return_value=_fake_result("hello")):
        reset_recorded_metrics()
        call_claude("prompt 1", "system", "sonnet")
        call_claude("prompt 2", "system", "sonnet")
        assert len(get_recorded_metrics()) == 2


def test_reset_clears_metrics():
    with patch("subprocess.run", return_value=_fake_result("hello")):
        reset_recorded_metrics()
        call_claude("prompt", "system", "sonnet")
        reset_recorded_metrics()
        assert get_recorded_metrics() == []


def test_missing_usage_defaults_to_zero():
    bad_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps({"result": "hello"})
    )
    with patch("subprocess.run", return_value=bad_result):
        reset_recorded_metrics()
        call_claude("prompt", "system", "sonnet")
        metrics = get_recorded_metrics()
        assert metrics[0].input_tokens == 0
        assert metrics[0].cost_usd == 0.0
