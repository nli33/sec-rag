"""Unit tests for secrag/baseline.py and secrag/eval/baseline_run.py.

No live `claude` CLI calls — subprocess.run is mocked.
"""
import json
import subprocess
from unittest.mock import patch

from secrag.baseline import answer_naive, answer_web_search
from secrag.claude_cli import reset_recorded_metrics
from secrag.eval.baseline_run import run_baseline_eval


def _fake_result(result_text: str) -> subprocess.CompletedProcess:
    payload = {
        "result": result_text,
        "total_cost_usd": 0.01,
        "duration_ms": 1000,
        "duration_api_ms": 900,
        "usage": {"input_tokens": 10, "output_tokens": 5,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload))


def test_answer_naive_passes_no_tools():
    with patch("subprocess.run", return_value=_fake_result("answer")) as mock_run:
        reset_recorded_metrics()
        assert answer_naive("What was revenue?") == "answer"
        args = mock_run.call_args[0][0]
        assert args[args.index("--tools") + 1] == ""


def test_answer_web_search_passes_websearch_tool():
    with patch("subprocess.run", return_value=_fake_result("answer")) as mock_run:
        reset_recorded_metrics()
        assert answer_web_search("What was revenue?") == "answer"
        args = mock_run.call_args[0][0]
        assert args[args.index("--tools") + 1] == "WebSearch"


def test_run_baseline_eval_scores_and_records_metrics():
    questions = [{
        "financebench_id": "q1", "question_type": "domain-relevant",
        "question": "What was FY2020 revenue?", "answer": "$100 million",
    }]
    with patch("subprocess.run", return_value=_fake_result("Revenue was $100 million.")):
        results = run_baseline_eval(questions, mode="naive")
    assert len(results) == 1
    assert results[0]["correct"] is True
    assert results[0]["evidence_recall"] is False
    assert results[0]["cost_usd"] == 0.01


def test_run_baseline_eval_records_error_on_failure():
    questions = [{
        "financebench_id": "q1", "question_type": "domain-relevant",
        "question": "What was FY2020 revenue?", "answer": "$100 million",
    }]
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["claude"], stderr="boom")):
        results = run_baseline_eval(questions, mode="naive")
    assert results[0]["correct"] is False
    assert results[0]["score_method"] == "error"
