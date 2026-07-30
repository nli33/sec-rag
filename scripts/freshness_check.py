"""Freshness check: does the RAG pipeline hold up on a filing the model can't have
memorized, and does the naive (no-tools) baseline collapse the way it should?

Gold answers here are pulled directly from XBRL facts (see notes/freshness-check-3m-q2-2026.md
for why), not hand-read/computed by a human or LLM — every question below is either a single
tagged XBRL fact or a ratio of two single-tagged facts, chosen specifically to avoid any
company-specific judgment call (e.g. quick ratio was deliberately excluded: 3M doesn't tag a
short-term-investments line at all, which would force a judgment call about whether that means
"the line is zero" or "we failed to find the right tag").

Uses a separate Qdrant collection (secrag_freshness_check) so this doesn't touch the validated
FinanceBench eval corpus.
"""
from secrag.baseline import answer_naive
from secrag.claude_cli import reset_recorded_metrics
from secrag.eval.run import _metrics_summary
from secrag.eval.scorer import score_answer
from secrag.generate import generate
from secrag.retrieve import retrieve

COLLECTION = "secrag_freshness_check"
TICKER = "MMM"

# Gold values extracted programmatically from 3M's actual Q2 2026 10-Q (filed 2026-07-21,
# 8 days before this check was built) via secrag.ingest.extract_facts's underlying XBRL data —
# see the note above for how these were pulled (period_instant/fiscal_period filtering, not
# manual reading).
QUESTIONS = [
    {
        "id": "revenue_q2_2026",
        "question": "What was 3M's total revenue for Q2 2026 (the three months ended June 30, 2026)?",
        "gold": "$6,500 million",
    },
    {
        "id": "net_income_q2_2026",
        "question": "What was 3M's net income for Q2 2026 (the three months ended June 30, 2026)?",
        "gold": "$933 million",
    },
    {
        "id": "accounts_receivable_q2_2026",
        "question": "What was 3M's accounts receivable, net, as of June 30, 2026?",
        "gold": "$3,927 million",
    },
    {
        "id": "inventory_q2_2026",
        "question": "What was 3M's inventory, net, as of June 30, 2026?",
        "gold": "$3,770 million",
    },
    {
        "id": "current_assets_q2_2026",
        "question": "What were 3M's total current assets as of June 30, 2026?",
        "gold": "$14,112 million",
    },
    {
        "id": "current_liabilities_q2_2026",
        "question": "What were 3M's total current liabilities as of June 30, 2026?",
        "gold": "$11,379 million",
    },
    {
        "id": "current_ratio_q2_2026",
        "question": "What was 3M's current ratio (total current assets divided by total current "
        "liabilities) as of June 30, 2026?",
        "gold": "1.24",
    },
]


def _run_one(mode: str, q: dict) -> dict:
    reset_recorded_metrics()
    import time

    t0 = time.time()
    try:
        if mode == "naive":
            answer = answer_naive(q["question"], company="3M Company")
            evidence_recall = False
        else:
            chunks = retrieve(q["question"], ticker=TICKER, collection_name=COLLECTION)
            answer = generate(q["question"], chunks)
            evidence_recall = len(chunks) > 0
        correct, method = score_answer(q["question"], q["gold"], answer)
        return {
            "id": q["id"], "mode": mode, "gold": q["gold"], "predicted": answer,
            "correct": correct, "score_method": method, "evidence_recall": evidence_recall,
            **_metrics_summary(time.time() - t0),
        }
    except Exception as e:
        return {
            "id": q["id"], "mode": mode, "gold": q["gold"], "predicted": None,
            "correct": False, "score_method": "error", "evidence_recall": False,
            **_metrics_summary(time.time() - t0), "error": str(e),
        }


def main():
    results = {"naive": [], "rag": []}
    for mode in ("naive", "rag"):
        for q in QUESTIONS:
            r = _run_one(mode, q)
            results[mode].append(r)
            print(f"[{mode}] {r['id']}: correct={r['correct']} predicted={str(r['predicted'])[:100]!r}")

    for mode in ("naive", "rag"):
        n_correct = sum(r["correct"] for r in results[mode])
        cost = sum(r["cost_usd"] for r in results[mode])
        print(f"\n{mode}: {n_correct}/{len(QUESTIONS)} correct, ${cost:.4f} total")

    return results


if __name__ == "__main__":
    main()
