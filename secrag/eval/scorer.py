"""M3: scoring — numeric normalizer, LLM-as-judge, evidence recall@k."""
import json
import re
from typing import Optional

from secrag.claude_cli import call_claude
from secrag.config import MODEL
from secrag.retrieve import RetrievedChunk

_SCALE_WORDS = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6,
    "billion": 1e9, "b": 1e9,
}

_NUMERIC_RE = re.compile(
    r"\(?-?(\$)?\s*(\d[\d,]*\.?\d*)\s*(thousand|million|billion|[kmb])?\b\)?%?",
    re.IGNORECASE,
)
# Strips our own citation format, e.g. "(ticker=3M_2018_10K, item=page_38)", which otherwise
# gets misread as a number (the "3" in "3M" parsed as digit + million-scale-word).
_CITATION_RE = re.compile(r"\([^()]*(?:ticker|item)=[^()]*\)")


def _numeric_candidates(text: str) -> list[tuple[float, bool]]:
    """Parse every $/comma/scale-word/percent/parenthesized-negative number in `text`.

    Returns a list of (value, had_explicit_scale_word) — every plausible reading, not just
    one "chosen" match. A single sentence/paragraph often contains several numbers (e.g. a
    model showing its work: intermediate dollar figures before a final percentage), and
    picking just one up front is fragile — see PERFORMANCE.md's accuracy diagnosis.
    """
    text = _CITATION_RE.sub("", text).strip()
    candidates = []
    for m in _NUMERIC_RE.finditer(text):
        scale = m.group(3)
        if scale and len(scale) == 1 and not m.group(1):
            # A bare single-letter scale abbreviation with no "$" prefix (e.g. the "M" in
            # "3M") is indistinguishable by regex from a ticker/proper noun ("3M" the
            # company) — every dollar figure in this codebase's generated/gold text is
            # either "$"-prefixed or spells the scale word out, so require "$" here rather
            # than risk parsing "3M" as 3,000,000.
            continue
        value = float(m.group(2).replace(",", ""))
        if scale:
            value *= _SCALE_WORDS[scale.lower()]
        raw = m.group(0).strip()
        if (
            not scale
            and "$" not in raw
            and "%" not in raw
            and value == int(value)
            and 1900 <= value <= 2100
        ):
            # A bare 4-digit integer with no scale/currency/percent marker, in a plausible
            # calendar-year range, is almost always a fiscal-year reference ("FY2023"), not
            # a real answer value — both gold and predicted text mention the fiscal year in
            # question, so treating it as a candidate answer causes coincidental false-positive
            # matches (e.g. gold "flat... FY 2023 vs FY 2022" matching any predicted text that
            # also happens to mention 2023).
            continue
        if raw.startswith("(") and raw.endswith(")"):
            value = -value
        candidates.append((value, bool(scale)))
    return candidates


def normalize_numeric(text: str) -> Optional[tuple[float, bool]]:
    """Parse a $/comma/scale-word/percent/parenthesized-negative numeric string to a float.

    Returns (value, had_explicit_scale_word) for the first candidate found, or None if no
    number is found. Gold answers are short and normally have exactly one real number, so
    "first candidate" is a reasonable single value to report; `numeric_match` below checks
    predicted text against *all* candidates instead, since predicted text is longer and the
    first number often isn't the final answer.
    """
    candidates = _numeric_candidates(text)
    return candidates[0] if candidates else None


def _question_scale_hint(question: str) -> Optional[float]:
    """FinanceBench gold answers are bare numbers in the unit the question asks for
    (e.g. "(in USD millions)"), unlike generated answers which spell out the scale word."""
    question = question.lower()
    for word in ("billion", "million", "thousand"):
        if word in question:
            return _SCALE_WORDS[word]
    return None


def numeric_match(predicted: str, gold: str, question: str = "", tol: float = 0.01) -> Optional[bool]:
    """Compare predicted vs gold as numbers within relative tolerance. None if gold isn't numeric.

    Checks predicted text's *every* candidate number against gold, not just the first one —
    a generated answer showing its work (e.g. "$116M / $6,489M = ... 1.9%") often states its
    real final answer well after earlier intermediate figures.
    """
    gold_parsed = normalize_numeric(gold)
    if gold_parsed is None:
        return None
    gold_val, gold_had_scale = gold_parsed
    if not gold_had_scale:
        hint = _question_scale_hint(question)
        if hint:
            gold_val *= hint

    pred_candidates = _numeric_candidates(predicted)
    if not pred_candidates:
        return False

    def _is_close(pred_val: float) -> bool:
        if gold_val == 0:
            return pred_val == 0
        return abs(pred_val - gold_val) / abs(gold_val) <= tol

    return any(_is_close(pred_val) for pred_val, _ in pred_candidates)


JUDGE_SYSTEM_PROMPT = """You are grading whether a candidate answer matches a gold answer to a
financial question. Respond with a single JSON object: {"correct": true|false}.
Mark correct if the candidate conveys the same fact as the gold answer, even if phrased
differently or with different precision. Mark incorrect if it states a different fact,
refuses, or is unsupported."""


def llm_judge(question: str, gold: str, predicted: str) -> bool:
    """Ask Claude (via CLI) whether `predicted` matches `gold` for `question`."""
    prompt = f"Question: {question}\nGold answer: {gold}\nCandidate answer: {predicted}"
    verdict = json.loads(call_claude(prompt, JUDGE_SYSTEM_PROMPT, MODEL))
    return bool(verdict["correct"])


def score_answer(question: str, gold: str, predicted: str) -> tuple[bool, str]:
    """Score `predicted` against `gold`. Returns (is_correct, method).

    A numeric *match* is trusted outright. A numeric *mismatch* is not trusted outright and
    falls back to the semantic judge instead — regex number extraction is inherently
    fallible (compound qualitative+numeric gold answers, narrative answers with incidental
    numbers), and a strict miss there doesn't mean the answer is actually wrong. See
    PERFORMANCE.md's accuracy diagnosis for confirmed cases this was masking.
    """
    is_numeric = numeric_match(predicted, gold, question)
    if is_numeric is True:
        return True, "numeric"
    if is_numeric is None:
        return llm_judge(question, gold, predicted), "llm_judge"
    return llm_judge(question, gold, predicted), "llm_judge_fallback"


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
