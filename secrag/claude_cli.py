"""Shared `claude` CLI invocation wrapper.

Every LLM call in this project shells out to the `claude` CLI rather than the
Anthropic API/SDK, so it rides the user's Claude Max subscription auth instead of
per-call API billing (see HANDOFF.md §3). This module centralizes that invocation
(previously duplicated near-identically in generate.py/calculator.py/decompose.py/
scorer.py) and records each call's token usage, cost, and duration — so efficiency
(not just correctness) can be measured, e.g. by secrag/eval/run.py per question.
"""
import json
import subprocess
from dataclasses import dataclass


@dataclass
class CallMetrics:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
    duration_ms: int
    duration_api_ms: int


# Module-level log of calls made since the last reset — a lightweight accumulator so
# callers (e.g. the eval harness) can measure a whole question's Claude usage (which may
# span several calls: decompose + generate + llm_judge) without threading a metrics object
# through every function signature. Reset before each unit of work you want to measure.
_recorded: list[CallMetrics] = []


def call_claude(prompt: str, system_prompt: str, model: str) -> str:
    """Run one `claude -p` call and return its raw "result" string.

    Raises subprocess.CalledProcessError / json.JSONDecodeError / KeyError on CLI or
    payload-shape failures — callers keep their own error handling for those exactly as
    before this was centralized (each caller's payload beyond "result" has different shape
    requirements, so payload-specific parsing/validation stays with the caller).
    """
    result = subprocess.run(
        [
            "claude",
            "-p", prompt,
            "--system-prompt", system_prompt,
            "--tools", "",
            "--disable-slash-commands",
            "--model", model,
            "--output-format", "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    usage = payload.get("usage", {})
    _recorded.append(
        CallMetrics(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            cost_usd=payload.get("total_cost_usd", 0.0),
            duration_ms=payload.get("duration_ms", 0),
            duration_api_ms=payload.get("duration_api_ms", 0),
        )
    )
    return payload["result"]


def reset_recorded_metrics() -> None:
    _recorded.clear()


def get_recorded_metrics() -> list[CallMetrics]:
    return list(_recorded)
