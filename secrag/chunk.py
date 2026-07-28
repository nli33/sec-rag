"""M2: split ingested sections into retrieval-sized chunks, inheriting provenance.

Chunks are sentence-aware (never split mid-sentence) and overlap by a couple of
sentences so a fact split across a chunk boundary is still findable from either side.
"""
import re
from dataclasses import dataclass

from secrag.ingest import Provenance, TextSection

# ~4 chars/token is a standard rule-of-thumb approximation (no tokenizer dependency needed).
CHARS_PER_TOKEN = 4
MAX_CHUNK_TOKENS = 450
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_SENTENCES = 2

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    provenance: Provenance
    text: str
    chunk_index: int


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _split_oversized(unit: str) -> list[str]:
    """Fall back to splitting a too-large sentence-splitter unit on newlines, then hard
    character windows as a last resort.

    Raw-text-extracted financial statement tables (balance sheets, income statements) have
    almost no sentence-ending punctuation, so the whole table becomes a single "sentence" to
    `_split_sentences` — the size cap in `chunk_section` only checks *between* sentences, so
    an entire table sails through as one oversized chunk. That dilutes any single line item's
    (e.g. "Inventory") signal across the whole table in one embedding vector, and was
    confirmed to cause a real retrieval miss (see notes/chunking-oversized-tables.md).

    A follow-up attempt to shrink this further (packing only ~2 rows per chunk instead of
    ~20) was tried and reverted — see notes/table-aware-chunking.md — it improved rerank
    concentration but regressed first-stage hybrid_search recall by stripping surrounding
    context (page/statement headers) that was helping chunks get found in the first place.
    """
    if len(unit) <= MAX_CHUNK_CHARS:
        return [unit]
    lines = [line for line in unit.split("\n") if line.strip()]
    if len(lines) <= 1:
        return [unit[i : i + MAX_CHUNK_CHARS] for i in range(0, len(unit), MAX_CHUNK_CHARS)]

    pieces = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if current and current_len + len(line) > MAX_CHUNK_CHARS:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_section(section: TextSection) -> list[Chunk]:
    """Greedily pack sentences into ~MAX_CHUNK_TOKENS windows with sentence overlap."""
    sentences = [unit for s in _split_sentences(section.text) for unit in _split_oversized(s)]
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def flush():
        if current:
            chunks.append(
                Chunk(section.provenance, " ".join(current), len(chunks))
            )

    for sentence in sentences:
        if current and current_len + len(sentence) > MAX_CHUNK_CHARS:
            flush()
            # Bound the carried-forward overlap by size, not just sentence count — a
            # `_split_oversized` fallback piece can itself be close to MAX_CHUNK_CHARS, and
            # blindly carrying the last OVERLAP_SENTENCES units forward regardless of their
            # size would let two large table pieces combine into a chunk far past the cap.
            overlap: list[str] = []
            overlap_len = 0
            for prev in reversed(current[-OVERLAP_SENTENCES:]):
                if overlap_len + len(prev) > MAX_CHUNK_CHARS // 4:
                    break
                overlap.insert(0, prev)
                overlap_len += len(prev)
            current, current_len = overlap, overlap_len
        current.append(sentence)
        current_len += len(sentence)
    flush()

    return chunks


def chunk_sections(sections: list[TextSection]) -> list[Chunk]:
    chunks = []
    for section in sections:
        chunks.extend(chunk_section(section))
    return chunks
