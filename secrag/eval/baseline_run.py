"""Baseline eval: same FinanceBench questions and scorer as eval/run.py, but the answer
comes from secrag.baseline (no ingest, no vector database, no retrieval over our indexed
filings) so the RAG pipeline's accuracy/cost can be compared against a baseline.
"""
import time

from secrag.baseline import answer_naive, answer_web_search
from secrag.claude_cli import reset_recorded_metrics
from secrag.eval.dataset import load_financebench
from secrag.eval.report import build_report
from secrag.eval.run import _metrics_summary
from secrag.eval.scorer import score_answer

MODES = {
    "naive": answer_naive,
    "web": answer_web_search,
}


def run_baseline_eval(questions: list[dict], mode: str = "web") -> list[dict]:
    """Answer+score every question with no retrieval over our indexed filings. Returns
    per-question results in the same shape as eval/run.py's run_eval (evidence_recall is
    always False — there's nothing from our pipeline to measure recall against)."""
    answer_fn = MODES[mode]
    results = []
    for q in questions:
        reset_recorded_metrics()
        t0 = time.time()
        try:
            answer = answer_fn(q["question"], company=q["company"])
            correct, method = score_answer(q["question"], q["answer"], answer)
            results.append({
                "financebench_id": q["financebench_id"],
                "question_type": q["question_type"],
                "gold": q["answer"],
                "predicted": answer,
                "correct": correct,
                "score_method": method,
                "evidence_recall": False,
                **_metrics_summary(time.time() - t0),
            })
        except Exception as e:
            results.append({
                "financebench_id": q["financebench_id"],
                "question_type": q["question_type"],
                "gold": q["answer"],
                "predicted": None,
                "correct": False,
                "score_method": "error",
                "evidence_recall": False,
                **_metrics_summary(time.time() - t0),
                "error": str(e),
            })
    return results


def main(mode: str = "web", limit: int = None):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    questions = load_financebench()
    if limit:
        questions = questions[:limit]
    results = run_baseline_eval(questions, mode=mode)
    print(build_report(results))
    return results


if __name__ == "__main__":
    main()
