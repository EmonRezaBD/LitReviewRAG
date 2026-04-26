"""Field definitions and prompt templates for structured paper extraction.

Defines the 8 target fields that the pipeline extracts from each paper,
each paired with a retrieval query (used by HybridSearcher) and a
description (used in the LLM extraction prompt).

The split between retrieval query and extraction prompt matters: the
query is optimized to pull relevant chunks from the paper, while the
prompt tells the LLM what to do with those chunks.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    """Specification for one extractable field.

    Attributes:
        name: Short snake_case identifier (used as JSON key + column name).
        retrieval_query: Natural-language query passed to HybridSearcher
            to find relevant chunks.
        description: Instruction shown to the LLM explaining what to
            extract for this field.
    """

    name: str
    retrieval_query: str
    description: str


# The 8 fields specified in the project proposal. Order is preserved in
# the Excel export, so keep title first for readability.
FIELD_SPECS: list[FieldSpec] = [
    FieldSpec(
        name="title",
        retrieval_query="paper title and authors",
        description=(
            "The full title of the paper as it appears at the top of the "
            "first page. Do not include author names or affiliations."
        ),
    ),
    FieldSpec(
        name="problem_statement",
        retrieval_query="What is the main problem or challenge this paper addresses?",
        description=(
            "A 1-3 sentence summary of the specific problem, gap, or "
            "challenge the paper addresses. Focus on what is wrong or "
            "missing in the current state of the field."
        ),
    ),
    FieldSpec(
        name="research_questions",
        retrieval_query="research questions hypotheses objectives",
        description=(
            "The explicit research questions, hypotheses, or objectives "
            "stated in the paper. List them as a single string separated "
            "by semicolons. If the paper does not state explicit research "
            "questions, summarize its implicit objectives."
        ),
    ),
    FieldSpec(
        name="contributions",
        retrieval_query="main contributions novel our contribution we propose",
        description=(
            "The paper's main contributions, novel ideas, or claimed "
            "advances. List 2-5 items as a single string separated by "
            "semicolons."
        ),
    ),
    FieldSpec(
        name="methodology",
        retrieval_query="methodology approach method experiment design implementation",
        description=(
            "The approach, techniques, models, datasets, or experimental "
            "setup the paper uses. 2-5 sentences. Be specific: name the "
            "algorithms, models, or datasets used."
        ),
    ),
    FieldSpec(
        name="findings",
        retrieval_query="results findings we found our results show accuracy",
        description=(
            "The paper's main empirical or theoretical findings. Include "
            "specific numbers, percentages, or effect sizes when reported. "
            "2-5 sentences."
        ),
    ),
    FieldSpec(
        name="limitations",
        retrieval_query="limitations weaknesses threats to validity caveats",
        description=(
            "Limitations, caveats, or threats to validity acknowledged by "
            "the authors. 1-3 sentences. If no limitations are explicitly "
            "stated, return null."
        ),
    ),
    FieldSpec(
        name="future_work",
        retrieval_query="future work directions next steps open questions",
        description=(
            "Future research directions or open problems proposed by the "
            "authors. 1-3 sentences. If no future work is stated, return null."
        ),
    ),
]


# Template for the extraction prompt. The LLM receives this with the
# field description and retrieved chunks substituted in. The strict JSON
# instruction and null-handling clause directly implement the proposal's
# bias-mitigation strategy (Section VII.B).
EXTRACTION_PROMPT_TEMPLATE = """You are extracting structured information from a research paper.

FIELD TO EXTRACT: {field_name}

FIELD DESCRIPTION:
{field_description}

RELEVANT PASSAGES FROM THE PAPER:
{context}

INSTRUCTIONS:
1. Extract ONLY the value for the field described above.
2. Base your answer strictly on the passages provided. Do not use outside knowledge.
3. If the passages do not contain enough information to answer, return null.
4. Do not invent or hallucinate information.
5. Respond with a JSON object matching this exact schema:
   {{"value": <string or null>}}

Respond with JSON only. No explanations, no markdown."""