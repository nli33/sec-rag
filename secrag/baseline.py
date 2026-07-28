"""Baselines for comparing against the RAG pipeline: no ingest, no vector database, no
retrieval over our indexed filings. Two variants, testing two different things:

- `answer_naive`: no tools at all, answers from the model's own training-data memory.
  Matches FinanceBench's own published "naive" baseline (~19% accuracy) — a cheap floor,
  but a weak test of this project's actual value-add, since any grounded system beats
  pure memory on filing-specific figures.
- `answer_web_search`: claude CLI with the WebSearch tool enabled, so it can find and read
  the actual filing itself (e.g. via SEC EDGAR or the company's investor relations site).
  This is the sharper comparison: it's the realistic alternative to this pipeline (a
  generic agent that can search the web), so if it matches or beats our custom
  chunking/retrieval/reranking machinery, that's a real signal the machinery isn't
  earning its complexity.
"""
from secrag.claude_cli import call_claude
from secrag.config import MODEL

NAIVE_SYSTEM_PROMPT = """You answer questions about a company's SEC filing using only your own
general knowledge. You do NOT have access to the actual filing document, the internet, or
any other retrieved source material.

Rules:
- Answer as best you can from what you already know. If you don't know or aren't
  confident of an exact figure, say so rather than guessing a precise number.
- Be concise and precise, especially with numbers.
"""

WEB_SEARCH_SYSTEM_PROMPT = """You answer questions about a company's SEC filing. You have
web search available — use it to find the actual filing (e.g. on SEC EDGAR or the
company's investor relations site) or other authoritative sources, rather than relying
solely on memory.

Rules:
- Prefer figures you can verify via search over recalling them from memory.
- If you can't find a reliable source for an exact figure, say so rather than guessing.
- Be concise and precise, especially with numbers.
"""


def answer_naive(question: str) -> str:
    """Answer `question` via the claude CLI with no tools and no retrieved context."""
    return call_claude(question, NAIVE_SYSTEM_PROMPT, MODEL, tools="")


def answer_web_search(question: str) -> str:
    """Answer `question` via the claude CLI with web search enabled, no retrieved context.

    Headless (-p) mode otherwise prompts for permission before a tool actually runs and
    silently no-ops it; bypassPermissions is scoped safely here by the narrow --tools
    WebSearch allowlist (a read-only search, nothing destructive).
    """
    return call_claude(
        question, WEB_SEARCH_SYSTEM_PROMPT, MODEL, tools="WebSearch", permission_mode="bypassPermissions"
    )
