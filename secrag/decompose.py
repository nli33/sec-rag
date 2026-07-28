"""Query decomposition: break a question into focused, independently-retrievable
sub-queries via the `claude` CLI (same pattern as generate.py/calculator.py).

Root-cause analysis of eval failures (see notes/retrieval-root-cause-analysis.md)
found the dominant remaining failure mode is questions needing two different facts
from two different filing pages (e.g. a ratio needs both a numerator and a
denominator, usually from different statements) — single-shot top-k retrieval
against the question's own wording often only surfaces one of the two pages needed.
"""
import json
import subprocess

from secrag.claude_cli import call_claude
from secrag.config import MODEL

SYSTEM_PROMPT = """You prepare a question about an SEC filing for retrieval against that
filing's text. Some questions need only one fact (e.g. "what was net income"); others
implicitly need two or more different facts from different parts of the filing (e.g. a
ratio needs both a numerator and a denominator, usually from different statements).

Respond with ONLY a JSON object: {"sub_queries": ["<query 1>", ...]}

Rules:
- If the question needs only one fact, respond with a single-element list containing a
  rewritten version of the question using standard financial-statement terminology (e.g.
  prefer "current assets", "inventory", "cost of revenue" over colloquial phrasing like
  "quick ratio" or "capital intensity") — this helps match retrieval to the filing's own
  vocabulary.
- If the question needs multiple distinct facts, respond with 2-4 sub-queries, each a
  short, standalone, retrievable statement of one specific fact needed (e.g. "total
  revenue for fiscal year 2019", "property, plant and equipment net value for fiscal
  year 2019").
- Do not answer the question. Only produce retrieval queries.
"""


MAX_SUB_QUERIES = 4


class DecomposeError(Exception):
    pass


def _parse_payload(raw: str) -> dict:
    """Parse Claude's {"sub_queries"} JSON payload, tolerating ```json fences.

    Raises DecomposeError (not a raw JSONDecodeError/KeyError) on malformed input.
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
        raise DecomposeError(f"decompose payload was not valid JSON: {raw!r}") from e
    if not isinstance(parsed, dict) or "sub_queries" not in parsed:
        raise DecomposeError(f"decompose payload missing sub_queries: {raw!r}")
    sub_queries = parsed["sub_queries"]
    if not isinstance(sub_queries, list) or not sub_queries or not all(
        isinstance(q, str) and q.strip() for q in sub_queries
    ):
        raise DecomposeError(f"decompose payload has invalid sub_queries: {raw!r}")
    return parsed


def decompose(question: str) -> list[str]:
    """Break `question` into focused sub-queries via the `claude` CLI.

    Fails open to `[question]` (i.e. no decomposition, today's behavior) on any expected
    failure — a decomposition hiccup should never make retrieval worse than the baseline.
    """
    try:
        raw = call_claude(question, SYSTEM_PROMPT, MODEL)
        parsed = _parse_payload(raw)
        return parsed["sub_queries"][:MAX_SUB_QUERIES]
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, DecomposeError):
        return [question]
