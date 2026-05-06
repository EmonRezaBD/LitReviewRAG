"""Unit tests verifying the field specifications are well-formed.

The 8 FIELD_SPECS are the contract the pipeline implements. Catching
typos and missing fields here prevents silent extraction failures.
"""

from litreviewrag.extraction.prompts import (
    EXTRACTION_PROMPT_TEMPLATE,
    FIELD_SPECS,
)


EXPECTED_FIELD_NAMES = {
    "title",
    "problem_statement",
    "research_questions",
    "contributions",
    "methodology",
    "findings",
    "limitations",
    "future_work",
}


def test_eight_field_specs_defined():
    """The proposal commits to exactly 8 fields per paper."""
    assert len(FIELD_SPECS) == 8


def test_all_expected_field_names_present():
    """Exact set of field names must match the proposal."""
    actual = {spec.name for spec in FIELD_SPECS}
    assert actual == EXPECTED_FIELD_NAMES


def test_each_field_has_nonempty_query_and_description():
    """Every spec must have a usable retrieval query and description."""
    for spec in FIELD_SPECS:
        assert spec.retrieval_query.strip(), f"{spec.name} missing retrieval_query"
        assert spec.description.strip(), f"{spec.name} missing description"


def test_extraction_prompt_template_has_required_placeholders():
    """The prompt template must accept the runtime substitutions we expect."""
    for placeholder in ("{field_name}", "{field_description}", "{context}"):
        assert placeholder in EXTRACTION_PROMPT_TEMPLATE, (
            f"Prompt template missing required placeholder: {placeholder}"
        )