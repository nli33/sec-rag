"""Shared configuration loaded from environment / .env."""
import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SEC_IDENTITY = os.environ.get("SEC_IDENTITY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
MODEL = "sonnet"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
# Use the INT8-quantized reranker (see scripts/quantize_reranker.py) instead of fp32.
# ~2.3x faster reranking, no confirmed accuracy regression on a 25-question eval sample —
# see notes/retrieval-performance.md. Off by default: not yet validated on the full eval set.
USE_INT8_RERANKER = os.environ.get("USE_INT8_RERANKER", "").lower() in ("1", "true", "yes")
INT8_RERANKER_PATH = os.path.join(DATA_DIR, "models", "bge-reranker-base-int8")
