"""M1: EDGAR ingestion — per-section filing text + core XBRL facts, with provenance.

Every text section and fact is tagged with a Provenance record so downstream
retrieval/generation can always cite back to {company, filing, section/concept}.
Raw extraction results are cached to disk per (ticker, accession) to avoid
re-hitting SEC EDGAR on repeated runs.
"""
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd
from edgar import Company, set_identity

from secrag.config import RAW_DIR, SEC_IDENTITY

# Bump when the extraction logic changes shape, so stale on-disk caches from
# an older version of this module are rebuilt instead of silently reused.
CACHE_VERSION = 1

# Logical concept -> ordered list of XBRL tags to try (concept drift: companies
# tag the same fact differently, e.g. post-ASC-606 filers use
# RevenueFromContractWithCustomerExcludingAssessedTax instead of Revenues).
CONCEPT_GROUPS = {
    "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
    "NetIncomeLoss": ["NetIncomeLoss"],
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "StockholdersEquity": ["StockholdersEquity"],
}

_identity_set = False


def _ensure_identity():
    global _identity_set
    if _identity_set:
        return
    if not SEC_IDENTITY:
        raise RuntimeError(
            "SEC_IDENTITY is not set. Add 'Your Name your.email@example.com' to .env "
            "(SEC EDGAR requires a descriptive User-Agent on every request)."
        )
    set_identity(SEC_IDENTITY)
    _identity_set = True


@dataclass
class Provenance:
    cik: str
    ticker: str
    accession: str
    form_type: str
    filing_date: str
    fiscal_period: Optional[str] = None
    item: Optional[str] = None  # e.g. "1A" — set for text sections
    concept: Optional[str] = None  # e.g. "NetIncomeLoss" — set for XBRL facts


@dataclass
class TextSection:
    provenance: Provenance
    text: str


@dataclass
class Fact:
    provenance: Provenance
    value: float
    unit: str
    period_start: Optional[str]
    period_end: Optional[str]


def _cache_path(ticker: str, accession: str, name: str) -> str:
    d = os.path.join(RAW_DIR, ticker.upper(), accession)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def get_latest_filing(ticker: str, form: str = "10-K"):
    """Fetch the most recent filing of `form` type for `ticker`."""
    _ensure_identity()
    company = Company(ticker)
    filing = company.get_filings(form=form).latest()
    if filing is None:
        raise ValueError(f"No {form} filings found for {ticker}")
    return filing


def _clean(value):
    """pandas leaves missing cells as NaN/NaT, which are truthy in Python (`nan or x` -> nan)."""
    return None if pd.isna(value) else value


def _base_provenance(ticker: str, filing) -> dict:
    return dict(
        cik=str(filing.cik),
        ticker=ticker.upper(),
        accession=filing.accession_no,
        form_type=filing.form,
        filing_date=str(filing.filing_date),
    )


def _read_cache(cache_file: str) -> Optional[list]:
    """Return cached records, or None if no cache exists or it's from an older logic version."""
    if not os.path.exists(cache_file):
        return None
    with open(cache_file) as f:
        payload = json.load(f)
    if payload.get("version") != CACHE_VERSION:
        return None
    return payload["items"]


def _write_cache(cache_file: str, items: list) -> None:
    with open(cache_file, "w") as f:
        json.dump({"version": CACHE_VERSION, "items": items}, f)


def extract_sections(ticker: str, filing) -> list[TextSection]:
    """Extract per-Item narrative text, tagged with provenance. Cached to disk."""
    cache_file = _cache_path(ticker, filing.accession_no, "sections.json")
    cached = _read_cache(cache_file)
    if cached is not None:
        return [TextSection(Provenance(**c["provenance"]), c["text"]) for c in cached]

    base = _base_provenance(ticker, filing)
    fiscal_period = str(filing.period_of_report) if filing.period_of_report else None
    obj = filing.obj()
    sections = []
    for section in obj.sections.values():
        if not section.item:
            continue  # boilerplate with no Item number, e.g. the signatures page
        text = section.text()
        if not text or not text.strip():
            continue
        prov = Provenance(**base, item=section.item, fiscal_period=fiscal_period)
        sections.append(TextSection(prov, text.strip()))

    _write_cache(cache_file, [{"provenance": asdict(s.provenance), "text": s.text} for s in sections])
    return sections


def extract_facts(ticker: str, filing) -> list[Fact]:
    """Extract core XBRL facts for this filing, tagged with provenance. Cached to disk."""
    cache_file = _cache_path(ticker, filing.accession_no, "facts.json")
    cached = _read_cache(cache_file)
    if cached is not None:
        return [
            Fact(Provenance(**c["provenance"]), c["value"], c["unit"], c["period_start"], c["period_end"])
            for c in cached
        ]

    base = _base_provenance(ticker, filing)
    facts_view = filing.xbrl().facts
    results = []
    for candidates in CONCEPT_GROUPS.values():
        for tag in candidates:
            qualified = f"us-gaap:{tag}"
            df = facts_view.query().by_concept(qualified).to_dataframe()
            if df is None or df.empty:
                continue
            # by_concept does partial matching and includes dimensioned (segment/member)
            # breakdowns — both must be filtered to get the single consolidated value.
            df = df[(df["concept"] == qualified) & (df["is_dimensioned"] == False)]  # noqa: E712
            if df.empty:
                continue
            for row in df.to_dict("records"):
                fiscal_year = _clean(row.get("fiscal_year"))
                period_start = _clean(row.get("period_start"))
                period_end = _clean(row.get("period_end")) or _clean(row.get("period_instant"))
                prov = Provenance(
                    **base,
                    concept=tag,
                    fiscal_period=str(int(fiscal_year)) if fiscal_year is not None else None,
                )
                results.append(
                    Fact(
                        prov,
                        float(row["numeric_value"]),
                        str(row.get("currency") or ""),
                        str(period_start) if period_start is not None else None,
                        str(period_end) if period_end is not None else None,
                    )
                )
            break  # found data for this logical concept; skip remaining fallback tags

    _write_cache(
        cache_file,
        [
            {
                "provenance": asdict(fct.provenance),
                "value": fct.value,
                "unit": fct.unit,
                "period_start": fct.period_start,
                "period_end": fct.period_end,
            }
            for fct in results
        ],
    )
    return results


def ingest(ticker: str, form: str = "10-K") -> tuple[list[TextSection], list[Fact]]:
    """Ingest one company's latest filing of `form` type: sections + facts."""
    filing = get_latest_filing(ticker, form)
    sections = extract_sections(ticker, filing)
    facts = extract_facts(ticker, filing)
    return sections, facts
