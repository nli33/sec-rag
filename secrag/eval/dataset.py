"""M3: load FinanceBench and fetch/extract/cache each source PDF's page text.

Each unique filing PDF is fetched once and cached to disk (raw PDF + extracted
per-page text), mirroring the disk-cache pattern in secrag/ingest.py.
"""
import json
import os

import fitz  # pymupdf
import requests
from datasets import load_dataset

from secrag.config import DATA_DIR, SEC_IDENTITY

CACHE_VERSION = 1
EVAL_CACHE_DIR = os.path.join(DATA_DIR, "eval")
PDF_DIR = os.path.join(EVAL_CACHE_DIR, "pdfs")
TEXT_DIR = os.path.join(EVAL_CACHE_DIR, "text")

# FinanceBench's own doc_link for these docs is dead (company-hosted investor-relations
# URLs rot — adobe.com, johnsonandjohnson.gcs-web.com, and s22.q4cdn.com all 404/DNS-fail/
# 451 as of this writing). SEC EDGAR is the stable, authoritative source for any public
# company's filings, so these are corrected EDGAR URLs for the same filing instead.
#
# Caveat: EDGAR serves these as HTML (10-Ks/8-Ks are filed as .htm, not .pdf, since the
# mid-2010s), which pymupdf paginates differently than the dataset's original PDF — so
# evidence_recall (which checks retrieved chunks against the dataset's PDF-based
# evidence_page_num) isn't reliably calibrated for these ~9 documents specifically.
# Answer-correctness scoring is unaffected, since that only checks the final answer text.
DEAD_LINK_OVERRIDES = {
    "ADOBE_2015_10K": "https://www.sec.gov/Archives/edgar/data/796343/000079634316000224/adbe10kfy15.htm",
    "ADOBE_2016_10K": "https://www.sec.gov/Archives/edgar/data/796343/000079634317000031/adbe10kfy16.htm",
    "ADOBE_2017_10K": "https://www.sec.gov/Archives/edgar/data/796343/000079634318000015/adbe10kfy17.htm",
    "ADOBE_2022_10K": "https://www.sec.gov/Archives/edgar/data/796343/000079634323000007/adbe-20221202.htm",
    "JOHNSON_JOHNSON_2022_10K": "https://www.sec.gov/Archives/edgar/data/200406/000020040623000016/jnj-20230101.htm",
    "JOHNSON_JOHNSON_2022Q4_EARNINGS": "https://www.sec.gov/Archives/edgar/data/200406/000020040623000005/a2022q4exhibit991.htm",
    "JOHNSON_JOHNSON_2023Q2_EARNINGS": "https://www.sec.gov/Archives/edgar/data/200406/000020040623000073/a2023q2exhibit991.htm",
    "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30": "https://www.sec.gov/Archives/edgar/data/200406/000020040623000091/jnj-20230830.htm",
    "MGMRESORTS_2022Q4_EARNINGS": "https://www.sec.gov/Archives/edgar/data/789570/000078957023000004/mgmexhibit991q42022er.htm",
}


def load_financebench() -> list[dict]:
    """Return the 150 FinanceBench questions as plain dicts."""
    ds = load_dataset("PatronusAI/financebench", split="train")
    return list(ds)


def _pdf_path(doc_name: str, ext: str = ".pdf") -> str:
    os.makedirs(PDF_DIR, exist_ok=True)
    return os.path.join(PDF_DIR, f"{doc_name}{ext}")


def _text_cache_path(doc_name: str) -> str:
    os.makedirs(TEXT_DIR, exist_ok=True)
    return os.path.join(TEXT_DIR, f"{doc_name}.json")


def fetch_pdf(doc_name: str, doc_link: str) -> str:
    """Download the filing PDF to disk if not already cached. Returns the local path.

    Uses DEAD_LINK_OVERRIDES's corrected EDGAR URL instead of doc_link when doc_name is a
    known-dead link — the cached file's extension follows the URL actually fetched (EDGAR
    overrides are .htm; everything else is .pdf) since pymupdf needs a matching extension
    to auto-detect format.
    """
    doc_link = DEAD_LINK_OVERRIDES.get(doc_name, doc_link)
    ext = os.path.splitext(doc_link)[1] or ".pdf"
    path = _pdf_path(doc_name, ext)
    if os.path.exists(path):
        return path
    headers = {"User-Agent": SEC_IDENTITY} if doc_name in DEAD_LINK_OVERRIDES else None
    resp = requests.get(doc_link, timeout=60, headers=headers)
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
