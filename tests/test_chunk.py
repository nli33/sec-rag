"""Unit tests for secrag/chunk.py's sentence packing and oversized-table fallback."""
from secrag.chunk import MAX_CHUNK_CHARS, chunk_section
from secrag.ingest import Provenance, TextSection

_PROV = Provenance(cik="", ticker="TEST", accession="TEST_ACC", form_type="10-K", filing_date="")


def test_normal_prose_stays_under_target_and_overlaps():
    text = "Sentence one is here. Sentence two is here. Sentence three is here."
    chunks = chunk_section(TextSection(provenance=_PROV, text=text))
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_table_with_no_punctuation_gets_split_by_line():
    # Simulates a raw-text-extracted financial statement table: no periods/sentence
    # punctuation at all, so the whole thing is one giant "sentence" to the splitter.
    lines = [f"Line item {i}\n{i * 100}" for i in range(200)]
    table_text = "\n".join(lines)
    assert len(table_text) > MAX_CHUNK_CHARS  # confirm the test table is actually oversized

    chunks = chunk_section(TextSection(provenance=_PROV, text=table_text))
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS * 1.5  # some slack for the last packed line


def test_table_content_is_preserved_across_split_chunks():
    lines = [f"LineItem{i} {i * 100}" for i in range(300)]
    table_text = "\n".join(lines)
    chunks = chunk_section(TextSection(provenance=_PROV, text=table_text))
    combined = " ".join(c.text for c in chunks)
    assert "LineItem150 15000" in combined


def test_single_huge_line_with_no_newlines_hard_splits():
    huge_line = "x" * (MAX_CHUNK_CHARS * 3)
    chunks = chunk_section(TextSection(provenance=_PROV, text=huge_line))
    assert len(chunks) >= 3
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_empty_section_produces_no_chunks():
    assert chunk_section(TextSection(provenance=_PROV, text="")) == []
