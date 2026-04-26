"""Cross-paper contradiction detection over extracted findings.

After all papers have been extracted, this module passes the findings
column of every paper to a synthesis LLM and asks it to identify factual
contradictions — places where two papers report incompatible empirical
results. Each contradiction cites the two source papers, satisfying the
proposal's verifiability requirement.
"""

import json
import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from litreviewrag import config
from litreviewrag.extraction.extractor import PaperExtraction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Contradiction:
    """A single factual contradiction detected between two papers.

    Attributes:
        paper_a: Filename of the first paper.
        paper_b: Filename of the second paper.
        claim_a: Specific claim or finding from paper A.
        claim_b: Specific contradictory claim from paper B.
        explanation: Brief explanation of why these claims contradict.
    """

    paper_a: str
    paper_b: str
    claim_a: str
    claim_b: str
    explanation: str


# Synthesis prompt. Asks the model to be conservative and only flag
# genuine factual contradictions, not stylistic or framing differences.
# This directly addresses the "false positive" risk identified in the
# Module 5 discussion post.
CONTRADICTION_PROMPT_TEMPLATE = """You are analyzing findings from multiple research papers to identify factual contradictions.

PAPERS AND THEIR FINDINGS:
{findings_block}

INSTRUCTIONS:
1. Identify up to {max_contradictions} factual contradictions between papers.
2. A "contradiction" means two papers report empirically incompatible results
   (different numbers, opposite conclusions, or mutually exclusive claims).
3. Do NOT flag differences in scope, framing, methodology, or domain. Papers
   studying different things are not contradicting each other.
4. For each contradiction, cite specific text from each paper's findings.
5. If you cannot find genuine contradictions, return an empty list.

Respond with a JSON object matching this exact schema:
{{
  "contradictions": [
    {{
      "paper_a": "<filename of first paper>",
      "paper_b": "<filename of second paper>",
      "claim_a": "<specific claim from paper A>",
      "claim_b": "<specific contradictory claim from paper B>",
      "explanation": "<1-2 sentence explanation of the contradiction>"
    }}
  ]
}}

Respond with JSON only. No explanations, no markdown."""


def _get_client() -> OpenAI:
    """Construct an OpenAI client using the configured API key.

    Returns:
        A configured OpenAI client instance.
    """
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _format_findings_block(extractions: list[PaperExtraction]) -> str:
    """Format each paper's findings into a labeled block for the prompt.

    Skips papers whose findings field is None (failed or empty extraction)
    so the synthesis model is not asked to compare against missing data.

    Args:
        extractions: Per-paper extraction results.

    Returns:
        A string with one labeled section per paper that has findings.
    """
    blocks = []
    for ex in extractions:
        findings = ex.fields.get("findings")
        if findings is None or findings.value is None:
            logger.warning("Skipping %s — no findings extracted", ex.paper_name)
            continue
        blocks.append(f"--- {ex.paper_name} ---\n{findings.value}")
    return "\n\n".join(blocks)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _call_llm(client: OpenAI, prompt: str, model: str) -> str:
    """Send the synthesis prompt to the LLM and return the raw response.

    Args:
        client: An initialized OpenAI client.
        prompt: The full synthesis prompt (already formatted).
        model: Model name (defaults to config.SYNTHESIS_MODEL).

    Returns:
        The model's raw response text (expected to be JSON).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,  # deterministic synthesis
    )
    return response.choices[0].message.content or ""


def _parse_contradictions(raw_response: str) -> list[Contradiction]:
    """Parse the LLM's JSON response into Contradiction objects.

    Args:
        raw_response: Raw JSON string from the LLM.

    Returns:
        List of Contradiction objects. Returns empty list on parse failure
        rather than raising, since downstream code (Excel export) should
        still produce output even if synthesis failed.
    """
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse contradiction JSON: %s | raw=%r", exc, raw_response[:300])
        return []

    items = parsed.get("contradictions", [])
    if not isinstance(items, list):
        logger.warning("'contradictions' is not a list: %r", items)
        return []

    contradictions: list[Contradiction] = []
    for item in items:
        try:
            contradictions.append(
                Contradiction(
                    paper_a=str(item["paper_a"]),
                    paper_b=str(item["paper_b"]),
                    claim_a=str(item["claim_a"]),
                    claim_b=str(item["claim_b"]),
                    explanation=str(item["explanation"]),
                )
            )
        except (KeyError, TypeError) as exc:
            # Skip malformed entries but keep going on the rest
            logger.warning("Skipping malformed contradiction entry: %s | %r", exc, item)
            continue
    return contradictions


def detect_contradictions(
    extractions: list[PaperExtraction],
    model: str | None = None,
    max_contradictions: int = 3,
) -> list[Contradiction]:
    """Detect factual contradictions across papers' findings.

    Args:
        extractions: All paper extractions to compare. Need at least 2 papers
            with non-null findings for any contradiction to be possible.
        model: LLM to use. Defaults to config.SYNTHESIS_MODEL (gpt-4o).
        max_contradictions: Cap on contradictions to return. Default 3 per
            the proposal specification.

    Returns:
        List of Contradiction objects (possibly empty if no genuine
        contradictions exist or fewer than 2 papers have findings).
    """
    model = model or config.SYNTHESIS_MODEL

    # Need at least 2 papers with findings for contradictions to exist
    papers_with_findings = [
        ex for ex in extractions
        if ex.fields.get("findings") and ex.fields["findings"].value
    ]
    if len(papers_with_findings) < 2:
        logger.info(
            "Need >=2 papers with findings; got %d. Skipping synthesis.",
            len(papers_with_findings),
        )
        return []

    findings_block = _format_findings_block(extractions)
    prompt = CONTRADICTION_PROMPT_TEMPLATE.format(
        findings_block=findings_block,
        max_contradictions=max_contradictions,
    )

    logger.info(
        "Detecting contradictions across %d papers using model=%s",
        len(papers_with_findings),
        model,
    )
    client = _get_client()
    raw = _call_llm(client, prompt, model)
    return _parse_contradictions(raw)