"""M4: program-of-thought calculator — Claude emits Python, we execute it sandboxed.

We shell out to the `claude` CLI (same pattern as generate.py/scorer.py — no SDK,
per HANDOFF.md §3) to have Claude write a short snippet that computes the answer,
then run that snippet ourselves in a subprocess sandbox rather than letting Claude
execute code directly. This gives every numeric answer a reproducible calc trace
(the code + its printed output).

Sandbox honesty note: this is a meaningful guardrail against accidental and casual
prompt-injected code (restricted builtins, import allowlist, CPU/time/output limits,
no filesystem/network builtins) — it is NOT a true security boundary. Deep reflection
tricks (e.g. `().__class__.__base__.__subclasses__()`) can still reach arbitrary
objects in a pure-Python sandbox like this one; real isolation would need OS-level
sandboxing (container/seccomp), which is out of scope for this local resume project.
"""
import json
import resource
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass

from secrag.claude_cli import call_claude
from secrag.config import MODEL
from secrag.generate import _build_context
from secrag.retrieve import RetrievedChunk

TIMEOUT_SECONDS = 5
CPU_SECONDS_LIMIT = 2  # below TIMEOUT_SECONDS so the CPU rlimit can actually fire first
MAX_OUTPUT_BYTES = 64 * 1024

_ALLOWED_MODULES = {"math", "decimal", "statistics"}

# Fixed runner script (no interpolation of untrusted code — it's piped in via stdin).
# Execs the untrusted snippet with an explicit safe-builtins allowlist and a globals dict
# that holds none of the runner's own names, so there's nothing like a live `_real_import`
# for generated code to grab onto (the escape found in the first cut of this module).
_RUNNER_SRC = """
import sys
import math, decimal, statistics

_ALLOWED_MODULES = {"math": math, "decimal": decimal, "statistics": statistics}

def _guarded_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed in the calculator sandbox")
    return _ALLOWED_MODULES[root]

_safe_builtins = {
    "__import__": _guarded_import,
    "print": print, "len": len, "range": range, "abs": abs, "round": round,
    "min": min, "max": max, "sum": sum, "sorted": sorted, "reversed": reversed,
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "frozenset": frozenset,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "divmod": divmod, "pow": pow, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError, "OverflowError": OverflowError,
    "Exception": Exception, "StopIteration": StopIteration,
}
# Deliberately excluded: open, eval, exec, compile, input, getattr, globals, vars, __import__
# (the real one), builtins module itself.

code = sys.stdin.read()
exec(compile(code, "<calculator>", "exec"), {"__builtins__": _safe_builtins})
"""

SYSTEM_PROMPT = """You answer numeric/derived financial questions using only the excerpts
provided below, by writing a short Python snippet that computes the exact answer.

Rules:
- Respond with ONLY a JSON object: {"code": "<python source>", "explanation": "<one line>"}
- The code must use decimal.Decimal for all money arithmetic (never float).
- The code's only imports may be from math, decimal, statistics.
- The code must end by printing the final numeric answer via print(...), nothing else.
- Use only values found in the excerpts; do not invent numbers.
"""


@dataclass
class CalcResult:
    answer: str
    code: str
    explanation: str


class CalculatorError(Exception):
    pass


def _limit_resources():
    # Runs in the forked child before exec; safe here since the reader thread below is
    # only started after Popen() (i.e. after the fork), so the parent is single-threaded
    # at fork time even though calculate() may be called repeatedly in a sequential loop.
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS_LIMIT, CPU_SECONDS_LIMIT))


def run_sandboxed(code: str) -> str:
    """Execute `code` in a sandboxed subprocess and return its stdout (stripped).

    Raises CalculatorError on a disallowed import/builtin, non-zero exit, timeout,
    or output exceeding MAX_OUTPUT_BYTES (the subprocess is killed in that case).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.Popen(
            [sys.executable, "-c", _RUNNER_SRC],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmpdir,
            env={},
            preexec_fn=_limit_resources,
        )
        proc.stdin.write(code)
        proc.stdin.close()

        stdout_chunks = []
        overflowed = False

        def _read_capped():
            nonlocal overflowed
            total = 0
            for chunk in iter(lambda: proc.stdout.read(4096), ""):
                total += len(chunk)
                if total > MAX_OUTPUT_BYTES:
                    overflowed = True
                    proc.kill()
                    return
                stdout_chunks.append(chunk)

        reader = threading.Thread(target=_read_capped, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            reader.join(timeout=1)
            raise CalculatorError(f"calculator code timed out after {TIMEOUT_SECONDS}s")

        reader.join(timeout=1)
        stderr = proc.stderr.read()

    if overflowed:
        raise CalculatorError(f"calculator code produced too much output (>{MAX_OUTPUT_BYTES} bytes)")
    if proc.returncode != 0:
        raise CalculatorError(f"calculator code failed: {stderr.strip()}")
    return "".join(stdout_chunks).strip()


def _parse_payload(raw: str) -> dict:
    """Parse Claude's {"code", "explanation"} JSON payload, tolerating ```json fences.

    Raises CalculatorError (not a raw JSONDecodeError/KeyError) on malformed input.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise CalculatorError(f"calculator payload was not valid JSON: {raw!r}") from e
    if not isinstance(parsed, dict) or "code" not in parsed or "explanation" not in parsed:
        raise CalculatorError(f"calculator payload missing code/explanation: {raw!r}")
    return parsed


def calculate(question: str, chunks: list[RetrievedChunk]) -> CalcResult:
    """Have Claude emit a Python snippet grounded in `chunks`, then execute it sandboxed."""
    context = _build_context(chunks)
    prompt = f"Excerpts:\n\n{context}\n\nQuestion: {question}"

    try:
        raw = call_claude(prompt, SYSTEM_PROMPT, MODEL)
    except (json.JSONDecodeError, KeyError) as e:
        raise CalculatorError(f"unexpected claude CLI output: {e}") from e

    parsed = _parse_payload(raw)
    code = parsed["code"]
    explanation = parsed["explanation"]

    answer = run_sandboxed(code)
    return CalcResult(answer=answer, code=code, explanation=explanation)
