"""M2: answer a question by shelling out to the `claude` CLI over retrieved chunks.

Uses the user's Claude Max subscription auth via the CLI rather than the
Anthropic API/SDK (avoids per-call API billing). See HANDOFF.md §3.
"""
import json
import subprocess

from secrag.config import MODEL
from secrag.retrieve import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about SEC filings using only the excerpts provided below.

Rules:
- Answer using only the given excerpts. If they don't contain the answer, say so.
- Cite the source of every fact using its (ticker, item) provenance shown with each excerpt.
- Be concise and precise, especially with numbers.
"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        ticker = chunk.provenance.get("ticker")
        item = chunk.provenance.get("item")
        parts.append(f"[ticker={ticker} item={item}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    """Answer `question` grounded in `chunks` via the `claude` CLI."""
    context = _build_context(chunks)
    prompt = f"Excerpts:\n\n{context}\n\nQuestion: {question}"

    result = subprocess.run(
        [
            "claude",
            "-p", prompt,
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",
            "--disable-slash-commands",
            "--model", MODEL,
            "--output-format", "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["result"]
