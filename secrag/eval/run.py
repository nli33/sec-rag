"""M3: run the FinanceBench eval — ingest each unique filing PDF once, then
retrieve+generate+score every question against it.
"""
import time

from qdrant_client import models

from secrag.chunk import chunk_sections
from secrag.claude_cli import get_recorded_metrics, reset_recorded_metrics
from secrag.eval.dataset import extract_pages, fetch_pdf, load_financebench
from secrag.eval.report import build_report
from secrag.eval.scorer import evidence_recall, score_answer
from secrag.generate import generate
from secrag.index import get_client, index_chunks
from secrag.ingest import Provenance, TextSection
from secrag.retrieve import retrieve

EVAL_COLLECTION = "secrag_eval_chunks"


def _doc_indexed(doc_name: str) -> bool:
    client = get_client()
    if not client.collection_exists(EVAL_COLLECTION):
        return False
    count = client.count(
        collection_name=EVAL_COLLECTION,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=doc_name.upper()))]
        ),
    ).count
    return count > 0


def ingest_doc(doc_name: str, doc_link: str, doc_type: str, doc_period) -> int:
    """Fetch, extract, chunk, and index one filing PDF. Skips if already indexed."""
    if _doc_indexed(doc_name):
        return 0

    pdf_path = fetch_pdf(doc_name, doc_link)
    pages = extract_pages(doc_name, pdf_path)

    sections = [
        TextSection(
            provenance=Provenance(
                cik="", ticker=doc_name.upper(), accession=doc_name,
                form_type=doc_type, filing_date=str(doc_period), item=f"page_{page_num}",
            ),
            text=text,
        )
        for page_num, text in pages
        if text.strip()
    ]
    chunks = chunk_sections(sections)
    return index_chunks(chunks, collection_name=EVAL_COLLECTION)


def _metrics_summary(seconds: float) -> dict:
    """Aggregate every `claude` CLI call recorded since the last reset into one summary —
    lets us track efficiency (tokens/cost/duration), not just correctness, per question."""
    metrics = get_recorded_metrics()
    return {
        "wall_time_s": seconds,
        "claude_calls": len(metrics),
        "input_tokens": sum(m.input_tokens for m in metrics),
        "output_tokens": sum(m.output_tokens for m in metrics),
        "cache_read_tokens": sum(m.cache_read_input_tokens for m in metrics),
        "cache_creation_tokens": sum(m.cache_creation_input_tokens for m in metrics),
        "cost_usd": sum(m.cost_usd for m in metrics),
    }


def run_eval(questions: list[dict]) -> list[dict]:
    """Ingest each question's doc (if needed), then retrieve+generate+score. Returns per-question results.

    A single question's failure (PDF fetch, retrieval, generation, or judging) is recorded and
    skipped rather than aborting the whole run — a 150-question pass shouldn't restart from
    scratch over one transient error.
    """
    results = []
    for q in questions:
        reset_recorded_metrics()
        t0 = time.time()
        try:
            ingest_doc(q["doc_name"], q["doc_link"], q["doc_type"], q["doc_period"])

            chunks = retrieve(q["question"], ticker=q["doc_name"], collection_name=EVAL_COLLECTION)
            answer = generate(q["question"], chunks)
            correct, method = score_answer(q["question"], q["answer"], answer)
            evidence_pages = [
                e["evidence_page_num"] for e in q["evidence"] if e.get("evidence_page_num") is not None
            ]

            results.append({
                "financebench_id": q["financebench_id"],
                "question_type": q["question_type"],
                "gold": q["answer"],
                "predicted": answer,
                "correct": correct,
                "score_method": method,
                "evidence_recall": evidence_recall(chunks, evidence_pages),
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


def main(limit: int = None):
    questions = load_financebench()
    if limit:
        questions = questions[:limit]
    results = run_eval(questions)
    print(build_report(results))
    return results


if __name__ == "__main__":
    main()
