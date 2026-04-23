"""Entry point for running LitReviewRAG as a module.

Enables the command: `python -m litreviewrag`
"""

from litreviewrag import config


def main() -> None:
    """Smoke test: validate config and print confirmation."""
    config.validate()
    print("✅ LitReviewRAG configured successfully.")
    print(f"   Extraction model: {config.EXTRACTION_MODEL}")
    print(f"   Embedding model:  {config.EMBEDDING_MODEL}")
    print(f"   ChromaDB path:    {config.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()