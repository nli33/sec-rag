"""Shared configuration loaded from environment / .env."""
import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SEC_IDENTITY = os.environ.get("SEC_IDENTITY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
