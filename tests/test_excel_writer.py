"""Unit tests for Excel-export formatting helpers.

Focus: bullet-point formatting logic that distinguishes list-style
fields (semicolon-separated) from prose fields (sentence-separated)
from plain prose fields (no transformation).
"""

from litreviewrag.export.excel_writer import _format_value


def test_plain_prose_field_returned_unchanged():
    """problem_statement is plain prose — no bullet transformation applied."""
    text = "This paper addresses gaps in LLM evaluation."
    assert _format_value("problem_statement", text) == text


def test_title_field_returned_unchanged():
    """title is short and prose; do not split or bullet."""
    text = "A Survey of Generative Agent Simulations"
    assert _format_value("title", text) == text


def test_semicolon_list_field_becomes_bulleted_lines():
    """contributions is a semicolon-separated list — split into bullets."""
    text = "First contribution; Second contribution; Third"
    out = _format_value("contributions", text)
    lines = out.split("\n")
    assert len(lines) == 3
    assert all(line.startswith("• ") for line in lines)
    assert "First contribution" in lines[0]


def test_prose_list_field_splits_on_sentence_boundaries():
    """methodology is prose — sentence-level split into bullets."""
    text = "First the authors did X. Then they evaluated Y. Finally they compared Z."
    out = _format_value("methodology", text)
    lines = out.split("\n")
    assert len(lines) == 3
    assert all(line.startswith("• ") for line in lines)


def test_empty_value_returns_empty_string():
    """None or empty input must yield empty output, not crash."""
    assert _format_value("contributions", "") == ""
    assert _format_value("findings", "") == ""


def test_decimal_in_prose_does_not_trigger_split():
    """Periods inside numbers like '85.3%' must not be split into separate bullets."""
    text = "The model achieved 85.3% accuracy. This was a significant result."
    out = _format_value("findings", text)
    lines = out.split("\n")
    # Should split into 2 sentences, not 3
    assert len(lines) == 2
    assert "85.3%" in out  # decimal preserved intact