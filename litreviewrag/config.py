"""Configuration loader for LitReviewRAG.

Loads environment variables from a .env file and exposes them as
module-level constants. Import from this module anywhere in the codebase
to access API keys, model names, and storage paths.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env into the process environment.
# This call is idempotent — safe to run multiple times.
load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------------------
# Model Names
# ---------------------------------------------------------------------------
EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
SYNTHESIS_MODEL: str = os.getenv("SYNTHESIS_MODEL", "gpt-4o")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BASELINE_MODEL: str = os.getenv(
    "BASELINE_MODEL", "meta-llama/llama-3.1-70b-instruct"
)

# ---------------------------------------------------------------------------
# Storage Paths
# ---------------------------------------------------------------------------
# ChromaDB persistent storage directory. Resolved to an absolute path so
# the CLI works regardless of the current working directory.
CHROMA_PERSIST_DIR: Path = Path(
    os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
).resolve()


def validate() -> None:
    """Fail fast if required environment variables are missing.

    Call this at application startup to surface configuration errors
    before the pipeline begins expensive work.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and "
            "add your OpenAI API key."
        )