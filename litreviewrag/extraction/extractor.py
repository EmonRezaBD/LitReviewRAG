"""Structured field extraction from retrieved paper chunks using GPT-4o-mini.

For each of the 8 target fields, this module:
1. Uses HybridSearcher to retrieve the top-k relevant chunks from one paper.
2. Formats them into the extraction prompt.
3. Calls GPT-4o-mini with JSON-mode response format enforcement.
4. Parses and returns the extracted value plus the source chunks for
   verification (citations, per the proposal's bias-mitigation plan).
"""

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from litreviewrag import config
from litreviewrag.extraction.prompts import (
    EXTRACTION_PROMPT_TEMPLATE,
    FIELD_SPECS,
    FieldSpec,
)
from litreviewrag.retrieval.hybrid_search import HybridResult, HybridSearcher

logger = logging.getLogger(__name__)


@dataclass
class FieldExtraction:
    """Result of extracting one field from one paper.

    Attributes:
        field_name: The field's short identifier (e.g., 'problem_statement').
        value: The extracted value, or None if the paper lacked the info.
        source_chunks: The retrieved chunks used as context. Retained so
            users can verify extractions, per the proposal's bias-mitigation
            plan (include source citations with every extracted field).
    """

    field_name: str
    value: str | None
    source_chunks: list[HybridResult] = field(default_factory=list)


@dataclass
class PaperExtraction:
    """All 8 field extractions for a single paper.

    Attributes:
        paper_name: Source PDF filename.
        fields: Mapping from field name to its FieldExtraction result.
    """

    paper_name: str
    fields: dict[str, FieldExtraction]


def _get_client() -> OpenAI:
    """Construct an OpenAI client using the configured API key.

    Returns:
        A configured OpenAI client instance.
    """
    return OpenAI(api_key=config.OPENAI_API_KEY)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _call_llm(client: OpenAI, prompt: str, model: str) -> str:
    """Send the extraction prompt to the LLM and return the raw response.

    Uses response_format={"type": "json_object"} so the model is
    constrained to valid JSON output, per the proposal's strict JSON
    output requirement (Section III.B).

    Args:
        client: An initialized OpenAI client.
        prompt: The full extraction prompt (already formatted).
        model: Model name (e.g., 'gpt-4o-mini').

    Returns:
        The model's raw response text (expected to be JSON).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,  # deterministic extraction
    )
    return response.choices[0].message.content or ""


def _parse_value(raw_response: str, field_name: str) -> str | None:
    """Parse the LLM's JSON response and extract the 'value' key.

    Args:
        raw_response: Raw JSON string from the LLM.
        field_name: Field name (used only for logging on parse failure).

    Returns:
        The string value, or None if the response contained null or
        could not be parsed. Parse failures are logged but do not raise,
        since one bad field should not abort the whole extraction.
    """
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse JSON for field '%s': %s | raw=%r",
            field_name,
            exc,
            raw_response[:200],
        )
        return None

    value = parsed.get("value")
    # Normalize: treat empty strings and the string "null" as None
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "null")):
        return None
    return str(value).strip()


def _format_context(chunks: list[HybridResult]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Args:
        chunks: Retrieved chunks, ordered by relevance.

    Returns:
        A string like "[Chunk 1]\\n...text...\\n\\n[Chunk 2]\\n..." ready
        to substitute into the extraction prompt.
    """
    parts = [f"[Chunk {i + 1}]\n{c.text}" for i, c in enumerate(chunks)]
    return "\n\n".join(parts)

def _get_first_chunks(
    searcher: HybridSearcher,
    paper_name: str,
    n: int = 2,
) -> list[HybridResult]:
    """Return the first n chunks of a paper in document order.

    Used for fields like 'title' that live in a fixed location at the
    start of the document. Bypasses similarity search entirely — the
    title is always near chunk_index 0, regardless of query phrasing.

    Args:
        searcher: The hybrid searcher (used only to access the underlying store).
        paper_name: Which paper to fetch from.
        n: Number of leading chunks to return.

    Returns:
        Up to n chunks ordered by chunk_index ascending. May be empty
        if the paper has no chunks in the store.
    """
    # Reach into the searcher's store to pull all chunks for this paper.
    # We then filter and sort by chunk_index — fast even for thousands of chunks.
    store = searcher._store  # noqa: SLF001 — internal accessor by design
    collection_records = store._collection.get(  # noqa: SLF001
        where={"paper_name": paper_name},
        include=["documents", "metadatas"],
    )
    docs = collection_records["documents"]
    metas = collection_records["metadatas"]
    if not docs:
        return []

    # Pair (chunk_index, text) and sort to get document-order chunks
    pairs = sorted(
        [
            (int(meta["chunk_index"]), doc, str(meta["paper_name"]))
            for doc, meta in zip(docs, metas)
        ],
        key=lambda x: x[0],
    )
    leading = pairs[:n]

    # Wrap in HybridResult so the caller's interface stays uniform.
    # bm25/vector scores are not meaningful here — this is positional retrieval.
    return [
        HybridResult(
            text=text,
            paper_name=p_name,
            chunk_index=idx,
            score=1.0,
            bm25_score=0.0,
            vector_score=0.0,
        )
        for idx, text, p_name in leading
    ]

def extract_field(
    searcher: HybridSearcher,
    paper_name: str,
    field_spec: FieldSpec,
    model: str | None = None,
    top_k: int = 5,
) -> FieldExtraction:
    """Extract one field from one paper.

    Args:
        searcher: A configured HybridSearcher over the vector store.
        paper_name: Which paper to extract from.
        field_spec: The field to extract.
        model: LLM to use. Defaults to config.EXTRACTION_MODEL.
        top_k: Number of chunks to retrieve as context.

    Returns:
        A FieldExtraction with the extracted value and source chunks.
    """
    model = model or config.EXTRACTION_MODEL

    # Special case: titles always live on the first page. RAG retrieval is
    # unreliable here because (a) the chunker often splits titles across
    # boundaries and (b) generic "title" queries also match cited papers
    # in the references. Use the document's first chunks directly.
    if field_spec.name == "title":
        chunks = _get_first_chunks(searcher, paper_name, n=2)
    else:
        chunks = searcher.search(
            query=field_spec.retrieval_query,
            paper_name=paper_name,
            top_k=top_k,
        )

    if not chunks:
        logger.warning(
            "No chunks retrieved for field '%s' in paper '%s'",
            field_spec.name,
            paper_name,
        )
        return FieldExtraction(field_name=field_spec.name, value=None)

    # Build the prompt
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        field_name=field_spec.name,
        field_description=field_spec.description,
        context=_format_context(chunks),
    )

    # Call the LLM
    client = _get_client()
    raw = _call_llm(client, prompt, model)

    # Parse and return
    value = _parse_value(raw, field_spec.name)
    return FieldExtraction(
        field_name=field_spec.name,
        value=value,
        source_chunks=chunks,
    )
    # 2. Build the prompt
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        field_name=field_spec.name,
        field_description=field_spec.description,
        context=_format_context(chunks),
    )

    # 3. Call the LLM
    client = _get_client()
    raw = _call_llm(client, prompt, model)

    # 4. Parse and return
    value = _parse_value(raw, field_spec.name)
    return FieldExtraction(
        field_name=field_spec.name,
        value=value,
        source_chunks=chunks,
    )


def extract_paper(
    searcher: HybridSearcher,
    paper_name: str,
    model: str | None = None,
    top_k: int = 5,
) -> PaperExtraction:
    """Extract all 8 fields from one paper.

    Args:
        searcher: A configured HybridSearcher over the vector store.
        paper_name: Which paper to extract from.
        model: LLM to use. Defaults to config.EXTRACTION_MODEL.
        top_k: Number of chunks to retrieve per field.

    Returns:
        A PaperExtraction containing results for all 8 fields.
    """
    logger.info("Extracting fields from %s using model=%s", paper_name, model)
    fields: dict[str, FieldExtraction] = {}
    for spec in FIELD_SPECS:
        fields[spec.name] = extract_field(
            searcher=searcher,
            paper_name=paper_name,
            field_spec=spec,
            model=model,
            top_k=top_k,
        )
    return PaperExtraction(paper_name=paper_name, fields=fields)