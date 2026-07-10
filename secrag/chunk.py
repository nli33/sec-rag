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


def chunk_section(section: TextSection) -> list[Chunk]:
    """Greedily pack sentences into ~MAX_CHUNK_TOKENS windows with sentence overlap."""
    sentences = _split_sentences(section.text)
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
            current = current[-OVERLAP_SENTENCES:]
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    flush()

    return chunks


def chunk_sections(sections: list[TextSection]) -> list[Chunk]:
    chunks = []
    for section in sections:
        chunks.extend(chunk_section(section))
    return chunks
