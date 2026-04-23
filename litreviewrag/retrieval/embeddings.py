"""OpenAI embedding client with batching and retry logic.

Wraps the OpenAI embeddings API to convert text chunks into dense vectors
suitable for similarity search. Uses batching to minimize API calls and
tenacity-based retries to handle transient rate limits and network errors.
"""

import logging
from typing import Iterable

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from litreviewrag import config

logger = logging.getLogger(__name__)

# OpenAI's embedding endpoint accepts up to 2048 inputs per request, but we
# use a smaller batch to keep individual requests under the token limit.
_BATCH_SIZE = 64


def _get_client() -> OpenAI:
    """Construct an OpenAI client using the configured API key.

    Returns:
        A configured OpenAI client instance.
    """
    return OpenAI(api_key=config.OPENAI_API_KEY)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _embed_batch(client: OpenAI, batch: list[str]) -> list[list[float]]:
    """Embed one batch of texts with exponential-backoff retries.

    Retries up to 5 times on any exception (rate limits, transient network
    errors) with exponential backoff (2s, 4s, 8s, 16s, 30s caps).

    Args:
        client: An initialized OpenAI client.
        batch: List of text strings to embed in a single API call.

    Returns:
        List of embedding vectors, one per input string, in the same order.
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=batch,
    )
    return [item.embedding for item in response.data]


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Embed a collection of text strings using OpenAI's embedding model.

    Automatically batches inputs to respect API limits and retries on
    transient failures. Input order is preserved in the output.

    Args:
        texts: Iterable of text strings to embed. Can be a list, generator,
            or any iterable.

    Returns:
        List of embedding vectors (each a list of floats), one per input
        string. Vector dimensionality depends on the configured model
        (1536 for text-embedding-3-small).
    """
    # Materialize once so we can batch and also report progress
    text_list = list(texts)
    if not text_list:
        return []

    client = _get_client()
    all_embeddings: list[list[float]] = []

    # Process in fixed-size batches to stay within per-request token limits
    for batch_start in range(0, len(text_list), _BATCH_SIZE):
        batch = text_list[batch_start : batch_start + _BATCH_SIZE]
        logger.info(
            "Embedding batch %d-%d of %d",
            batch_start,
            batch_start + len(batch),
            len(text_list),
        )
        batch_vectors = _embed_batch(client, batch)
        all_embeddings.extend(batch_vectors)

    return all_embeddings