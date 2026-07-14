"""M3: scoring — numeric normalizer, LLM-as-judge, evidence recall@k."""
import json
import re
import subprocess
from typing import Optional

from secrag.retrieve import RetrievedChunk

_SCALE_WORDS = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6,
    "billion": 1e9, "b": 1e9,
}

_NUMERIC_RE = re.compile(
    r"\(?-?(\$)?\s*(\d[\d,]*\.?\d*)\s*(thousand|million|billion|[kmb])?\)?%?",
    re.IGNORECASE,
)
# Strips our own citation format, e.g. "(ticker=3M_2018_10K, item=page_38)", which otherwise
# gets misread as a number (the "3" in "3M" parsed as digit + million-scale-word).
_CITATION_RE = re.compile(r"\([^()]*(?:ticker|item)=[^()]*\)")


def normalize_numeric(text: str) -> Optional[tuple[float, bool]]:
    """Parse a $/comma/scale-word/percent/parenthesized-negative numeric string to a float.

    Returns (value, had_explicit_scale_word) or None if no number is found. When multiple
    numbers appear (as in a full-sentence answer), prefers a dollar-sign-prefixed match.
    """
    text = _CITATION_RE.sub("", text).strip()
    matches = list(_NUMERIC_RE.finditer(text))
    if not matches:
        return None
    match = next((m for m in matches if m.group(1)), matches[0])

    value = float(match.group(2).replace(",", ""))
    scale = match.group(3)
    if scale:
        value *= _SCALE_WORDS[scale.lower()]
    if match.group(0).strip().startswith("(") and match.group(0).strip().endswith(")"):
        value = -value
    return value, bool(scale)


def _question_scale_hint(question: str) -> Optional[float]:
    """FinanceBench gold answers are bare numbers in the unit the question asks for
    (e.g. "(in USD millions)"), unlike generated answers which spell out the scale word."""
    question = question.lower()
    for word in ("billion", "million", "thousand"):
        if word in question:
            return _SCALE_WORDS[word]
    return None


def numeric_match(predicted: str, gold: str, question: str = "", tol: float = 0.01) -> Optional[bool]:
    """Compare predicted vs gold as numbers within relative tolerance. None if gold isn't numeric."""
    gold_parsed = normalize_numeric(gold)
    if gold_parsed is None:
        return None
    gold_val, gold_had_scale = gold_parsed
    if not gold_had_scale:
        hint = _question_scale_hint(question)
        if hint:
            gold_val *= hint

    pred_parsed = normalize_numeric(predicted)
    if pred_parsed is None:
        return False
    pred_val, _ = pred_parsed

    if gold_val == 0:
        return pred_val == 0
    return abs(pred_val - gold_val) / abs(gold_val) <= tol


JUDGE_SYSTEM_PROMPT = """You are grading whether a candidate answer matches a gold answer to a
financial question. Respond with a single JSON object: {"correct": true|false}.
Mark correct if the candidate conveys the same fact as the gold answer, even if phrased
differently or with different precision. Mark incorrect if it states a different fact,
refuses, or is unsupported."""


def llm_judge(question: str, gold: str, predicted: str) -> bool:
    """Ask Claude (via CLI) whether `predicted` matches `gold` for `question`."""
    prompt = f"Question: {question}\nGold answer: {gold}\nCandidate answer: {predicted}"
    result = subprocess.run(
        [
            "claude",
            "-p", prompt,
            "--system-prompt", JUDGE_SYSTEM_PROMPT,
            "--tools", "",
            "--disable-slash-commands",
            "--model", "sonnet",
            "--output-format", "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    verdict = json.loads(json.loads(result.stdout)["result"])
    return bool(verdict["correct"])


def score_answer(question: str, gold: str, predicted: str) -> tuple[bool, str]:
    """Score `predicted` against `gold`. Returns (is_correct, method)."""
    is_numeric = numeric_match(predicted, gold, question)
    if is_numeric is not None:
        return is_numeric, "numeric"
    return llm_judge(question, gold, predicted), "llm_judge"


def evidence_recall(retrieved: list[RetrievedChunk], evidence_pages: list[int]) -> bool:
    """Did retrieval surface any chunk from a gold evidence page?"""
    if not evidence_pages:
        return True
    retrieved_pages = set()
    for chunk in retrieved:
        item = chunk.provenance.get("item", "")
        if item.startswith("page_"):
            retrieved_pages.add(int(item.removeprefix("page_")))
    return bool(retrieved_pages & set(evidence_pages))
