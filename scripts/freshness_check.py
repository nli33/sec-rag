"""Freshness check: does the RAG pipeline hold up on filings the model can't have
memorized, and does the naive (no-tools) baseline collapse the way it should?

Gold answers here are pulled directly from XBRL facts (see
notes/freshness-check-3m-q2-2026.md for why), not hand-read/computed by a human or LLM —
every question is either a single tagged XBRL fact or a ratio of two single-tagged facts,
per company, chosen specifically to avoid any company-specific judgment call. A concept
missing for a given company (e.g. Adobe/Amcor don't tag InventoryNet the way MMM/AMZN/AES
do) is skipped for that company rather than forced — no attempt to decide "is this really
zero or did we miss the tag," which would reintroduce the judgment call this design avoids.
Quick ratio is excluded entirely: no company checked here cleanly tags a separate
short-term-investments line, so it can't be computed without that same judgment call.

Uses a separate Qdrant collection (secrag_freshness_check) so this doesn't touch the
validated FinanceBench eval corpus. Every company here is already in that corpus (older
filings) — this only adds each company's newest (post-training-cutoff) filing.
"""
import time

from secrag.baseline import answer_naive
from secrag.chunk import chunk_sections
from secrag.claude_cli import reset_recorded_metrics
from secrag.eval.run import _metrics_summary
from secrag.eval.scorer import score_answer
from secrag.generate import generate
from secrag.index import get_client, index_chunks
from secrag.ingest import get_latest_filing, extract_sections
from secrag.retrieve import retrieve
from qdrant_client import models

COLLECTION = "secrag_freshness_check"

# ticker -> (company display name, form type to fetch)
COMPANIES = {
    "MMM": "3M Company",
    "AMZN": "Amazon.com, Inc.",
    "ADBE": "Adobe Inc.",
    "AES": "The AES Corporation",
    "AMCR": "Amcor plc",
}

# concept key -> (question template, ordered list of alias XBRL tags to try, is a duration
# (Q-over-Q flow) fact rather than an instant (point-in-time balance) fact)
CONCEPTS = {
    "revenue": (
        "What was {company}'s total revenue for the quarter ended {period_end}?",
        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
        True,
    ),
    "net_income": (
        "What was {company}'s net income for the quarter ended {period_end}?",
        ["NetIncomeLoss"],
        True,
    ),
    "accounts_receivable": (
        "What was {company}'s accounts receivable, net, as of {period_end}?",
        ["AccountsReceivableNetCurrent"],
        False,
    ),
    "inventory": (
        "What was {company}'s inventory, net, as of {period_end}?",
        ["InventoryNet"],
        False,
    ),
    "current_assets": (
        "What were {company}'s total current assets as of {period_end}?",
        ["AssetsCurrent"],
        False,
    ),
    "current_liabilities": (
        "What were {company}'s total current liabilities as of {period_end}?",
        ["LiabilitiesCurrent"],
        False,
    ),
}


def _fmt_millions(value: float) -> str:
    return f"${value / 1e6:,.0f} million"


def _resolve_instant(df, aliases, period_end):
    for alias in aliases:
        sub = df[
            (df["concept"] == "us-gaap:" + alias)
            & (df["is_dimensioned"] == False)  # noqa: E712
            & (df["period_instant"] == period_end)
        ]
        if len(sub):
            return sub["numeric_value"].iloc[0]
    return None


def _resolve_quarter_duration(df, aliases, period_start, period_end, fiscal_year):
    for alias in aliases:
        sub = df[
            (df["concept"] == "us-gaap:" + alias)
            & (df["is_dimensioned"] == False)  # noqa: E712
            & (df["period_start"] == period_start)
            & (df["period_end"] == period_end)
            & (df["fiscal_year"] == fiscal_year)
        ]
        if len(sub):
            return sub["numeric_value"].iloc[0]
    return None


def build_questions_for_company(ticker: str, company: str) -> tuple[list[dict], object]:
    """Ingest `ticker`'s latest 10-Q and derive its question set from XBRL facts. Returns
    (questions, filing) — filing is returned so the caller can log what was actually fetched."""
    filing = get_latest_filing(ticker, form="10-Q")
    df = filing.xbrl().facts.to_dataframe()
    period_end = str(filing.period_of_report)

    # The quarter's duration window is the 3-month period ending period_end, in the same
    # fiscal year — NOT the YTD-cumulative period also present in the same facts table.
    quarter_starts = sorted(df[df["period_end"] == period_end]["period_start"].dropna().unique())
    period_start = quarter_starts[-1] if quarter_starts else None
    fiscal_year = int(period_end[:4])

    values = {}
    for key, (template, aliases, is_duration) in CONCEPTS.items():
        if is_duration:
            v = _resolve_quarter_duration(df, aliases, period_start, period_end, fiscal_year)
        else:
            v = _resolve_instant(df, aliases, period_end)
        if v is not None:
            values[key] = v

    questions = [
        {
            "id": f"{ticker}_{key}",
            "question": template.format(company=company, period_end=period_end),
            "gold": _fmt_millions(values[key]),
        }
        for key, (template, _, _) in CONCEPTS.items()
        if key in values
    ]

    if "current_assets" in values and "current_liabilities" in values:
        ratio = values["current_assets"] / values["current_liabilities"]
        questions.append({
            "id": f"{ticker}_current_ratio",
            "question": (
                f"What was {company}'s current ratio (total current assets divided by "
                f"total current liabilities) as of {period_end}?"
            ),
            "gold": f"{ratio:.2f}",
        })

    return questions, filing


def ingest_company(ticker: str, filing) -> int:
    client = get_client()
    existing = client.count(
        COLLECTION,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))]
        ),
    ).count if client.collection_exists(COLLECTION) else 0
    if existing:
        return 0
    sections = extract_sections(ticker, filing)
    chunks = chunk_sections(sections)
    return index_chunks(chunks, collection_name=COLLECTION)


def _run_one(mode: str, ticker: str, company: str, q: dict) -> dict:
    reset_recorded_metrics()
    t0 = time.time()
    try:
        if mode == "naive":
            answer = answer_naive(q["question"], company=company)
            evidence_recall = False
        else:
            chunks = retrieve(q["question"], ticker=ticker, collection_name=COLLECTION)
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
    all_questions = []  # list of (ticker, company, question dict)
    for ticker, company in COMPANIES.items():
        questions, filing = build_questions_for_company(ticker, company)
        print(f"{ticker}: filed {filing.filing_date}, {len(questions)} questions derivable")
        n_indexed = ingest_company(ticker, filing)
        print(f"{ticker}: indexed {n_indexed} chunks")
        for q in questions:
            all_questions.append((ticker, company, q))

    results = {"naive": [], "rag": []}
    for mode in ("naive", "rag"):
        for ticker, company, q in all_questions:
            r = _run_one(mode, ticker, company, q)
            results[mode].append(r)
            print(f"[{mode}] {r['id']}: correct={r['correct']} predicted={str(r['predicted'])[:100]!r}")

    for mode in ("naive", "rag"):
        n_correct = sum(r["correct"] for r in results[mode])
        cost = sum(r["cost_usd"] for r in results[mode])
        print(f"\n{mode}: {n_correct}/{len(all_questions)} correct, ${cost:.4f} total")

    return results


if __name__ == "__main__":
    main()
