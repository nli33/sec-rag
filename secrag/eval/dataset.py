"""M3: load FinanceBench and fetch/extract/cache each source PDF's page text.

Each unique filing PDF is fetched once and cached to disk (raw PDF + extracted
per-page text), mirroring the disk-cache pattern in secrag/ingest.py.
"""
import json
import os

import fitz  # pymupdf
import requests
from datasets import load_dataset

from secrag.config import DATA_DIR

CACHE_VERSION = 1
EVAL_CACHE_DIR = os.path.join(DATA_DIR, "eval")
PDF_DIR = os.path.join(EVAL_CACHE_DIR, "pdfs")
TEXT_DIR = os.path.join(EVAL_CACHE_DIR, "text")


def load_financebench() -> list[dict]:
    """Return the 150 FinanceBench questions as plain dicts."""
    ds = load_dataset("PatronusAI/financebench", split="train")
    return list(ds)


def _pdf_path(doc_name: str) -> str:
    os.makedirs(PDF_DIR, exist_ok=True)
    return os.path.join(PDF_DIR, f"{doc_name}.pdf")


def _text_cache_path(doc_name: str) -> str:
    os.makedirs(TEXT_DIR, exist_ok=True)
    return os.path.join(TEXT_DIR, f"{doc_name}.json")


def fetch_pdf(doc_name: str, doc_link: str) -> str:
    """Download the filing PDF to disk if not already cached. Returns the local path."""
    path = _pdf_path(doc_name)
    if os.path.exists(path):
        return path
    resp = requests.get(doc_link, timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def extract_pages(doc_name: str, pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_num, text), ...] for a PDF, cached to disk (page_num is 0-indexed)."""
    cache_path = _text_cache_path(doc_name)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            payload = json.load(f)
        if payload.get("version") == CACHE_VERSION:
            return [(p["page_num"], p["text"]) for p in payload["pages"]]

    doc = fitz.open(pdf_path)
    pages = [(i, page.get_text()) for i, page in enumerate(doc)]
    doc.close()

    with open(cache_path, "w") as f:
        json.dump(
            {"version": CACHE_VERSION, "pages": [{"page_num": n, "text": t} for n, t in pages]},
            f,
        )
    return pages
