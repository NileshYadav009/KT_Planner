"""Tests for pdf_rendering.py's rendering helpers — _render_inline_text (the
literal-markdown fix, REPOSITORY_AUDIT.md §9m) and render_section_blocks'
table-class tagging / title-dedup / warning-card scoping (§9l/§9m).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pdf_rendering import _render_inline_text, render_section_blocks, build_rendered_sections
from renderers.blocks.common import NOT_COVERED_MESSAGE, no_coverage_block


def test_render_inline_text_converts_bold_markdown():
    assert _render_inline_text("**Deployment Steps**") == "<strong>Deployment Steps</strong>"


def test_render_inline_text_converts_bold_within_surrounding_text():
    result = _render_inline_text("**Rollback trigger:** Deployment failure")
    assert result == "<strong>Rollback trigger:</strong> Deployment failure"


def test_render_inline_text_leaves_plain_text_unchanged():
    assert _render_inline_text("Plain text, no markdown") == "Plain text, no markdown"


def test_render_inline_text_escapes_html_special_characters():
    result = _render_inline_text('<script>alert(1)</script>')
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_render_inline_text_escapes_and_bolds_together():
    result = _render_inline_text('a & b "quoted" **bold**')
    assert result == 'a &amp; b &quot;quoted&quot; <strong>bold</strong>'


def test_render_inline_text_handles_none():
    assert _render_inline_text(None) == ""


def test_render_inline_text_converts_italic_markdown():
    # knowledge_builder.py's INFERRED_MARKER (" *(inferred...)*") relies on
    # this so a genuinely-inferred field value reads as visually distinct.
    result = _render_inline_text("High *(inferred — not explicitly stated in the transcript)*")
    assert "<em>(inferred — not explicitly stated in the transcript)</em>" in result
    assert "*" not in result


def test_render_inline_text_bold_and_italic_together():
    result = _render_inline_text("**Bold** and *italic*")
    assert result == "<strong>Bold</strong> and <em>italic</em>"


def test_no_coverage_block_uses_the_shared_message():
    block = no_coverage_block("Some Section")
    assert block["type"] == "NarrativeBlock"
    assert block["paragraphs"] == [NOT_COVERED_MESSAGE]


def _section(section_id, title, blocks):
    return {"section_id": section_id, "section_title": title, "blocks": blocks}


def test_render_section_blocks_tags_kv_table_for_technology_grid():
    sections = [_section("a", "A", [
        {"type": "TechnologyGrid", "title": "A", "rows": [{"label": "x", "value": "y"}]},
    ])]
    html = render_section_blocks(sections)
    assert 'class="kv-table"' in html
    assert 'class="grid-table"' not in html


def test_render_section_blocks_tags_kv_table_for_ownership_table():
    sections = [_section("a", "A", [
        {"type": "OwnershipTable", "title": "A", "rows": [{"role": "x", "team": "y"}]},
    ])]
    html = render_section_blocks(sections)
    assert 'class="kv-table"' in html


def test_render_section_blocks_tags_grid_table_for_decision_table():
    sections = [_section("a", "A", [
        {"type": "DecisionTable", "title": "A", "columns": ["C1", "C2"], "rows": [{"C1": "x", "C2": "y"}]},
    ])]
    html = render_section_blocks(sections)
    assert 'class="grid-table"' in html
    assert 'class="kv-table"' not in html


def test_render_section_blocks_skips_duplicate_block_title():
    sections = [_section("a", "System Overview", [
        {"type": "NarrativeBlock", "title": "System Overview", "paragraphs": ["Text."]},
    ])]
    html = render_section_blocks(sections)
    assert html.count("System Overview") == 1


def test_render_section_blocks_keeps_distinct_block_title():
    sections = [_section("a", "Disaster Recovery", [
        {"type": "ChecklistBlock", "title": "Recovery actions", "items": ["Restore snapshots"]},
    ])]
    html = render_section_blocks(sections)
    assert "Disaster Recovery" in html
    assert "Recovery actions" in html


def test_render_section_blocks_only_warning_blocks_get_warning_card_class():
    sections = [_section("a", "A", [
        {"type": "WarningBlock", "title": "A", "warnings": ["Do not touch this."]},
        {"type": "NarrativeBlock", "title": "Notes", "paragraphs": ["**Bold** narrative text."]},
    ])]
    html = render_section_blocks(sections)
    assert 'class="block-card warning-card"' in html
    # The NarrativeBlock's markdown bold must not trigger warning styling —
    # this is the exact bug the old :has(p strong) CSS selector caused.
    assert html.count("warning-card") == 1


def test_render_section_blocks_no_literal_bold_markdown_in_tables():
    sections = [_section("a", "A", [
        {"type": "DecisionTable", "title": "A", "columns": ["Col"], "rows": [{"Col": "**bold value**"}]},
    ])]
    html = render_section_blocks(sections)
    assert "**" not in html
    assert "<strong>bold value</strong>" in html


def test_render_section_blocks_numbers_sections_sequentially():
    sections = [
        _section("a", "First", [{"type": "NarrativeBlock", "title": "x", "paragraphs": ["p"]}]),
        _section("b", "Second", [{"type": "NarrativeBlock", "title": "y", "paragraphs": ["p"]}]),
    ]
    html = render_section_blocks(sections)
    assert '<span class="section-number">01</span>' in html
    assert '<span class="section-number">02</span>' in html


def test_decision_table_blank_cell_shows_not_covered_marker():
    sections = [_section("a", "A", [
        {"type": "DecisionTable", "title": "A", "columns": ["Symptom", "Fix"],
         "rows": [{"Symptom": "Pool exhaustion", "Fix": ""}]},
    ])]
    html = render_section_blocks(sections)
    assert '<td class="cell-not-covered">Not covered during KT</td>' in html
    assert "<td></td>" not in html


def test_decision_table_populated_cell_unaffected():
    sections = [_section("a", "A", [
        {"type": "DecisionTable", "title": "A", "columns": ["Symptom", "Fix"],
         "rows": [{"Symptom": "Pool exhaustion", "Fix": "Scale connections"}]},
    ])]
    html = render_section_blocks(sections)
    assert "Scale connections" in html
    assert "cell-not-covered" not in html


def _rendered_section(section, **overrides):
    base = {
        "id": section, "title": section, "status": "missing", "confidence": 0.0, "risk": 1.0,
        "facts": [], "entities": [], "evidence": [], "relationships": [], "fields": {},
        "coverage_content": [],
    }
    base.update(overrides)
    return base


def test_build_rendered_sections_omits_empty_conditional_section():
    knowledge_object = {"sections": [
        _rendered_section("first_30_day_ownership", title="FIRST 30-DAY OWNERSHIP PLAN", tier="conditional"),
    ]}
    rendered = build_rendered_sections(knowledge_object)
    assert rendered == []


def test_build_rendered_sections_keeps_empty_core_section():
    knowledge_object = {"sections": [
        _rendered_section("ownership_escalation", title="OWNERSHIP & ESCALATION"),
    ]}
    rendered = build_rendered_sections(knowledge_object)
    assert len(rendered) == 1
    assert rendered[0]["section_id"] == "ownership_escalation"


def test_build_rendered_sections_keeps_conditional_section_with_real_content():
    knowledge_object = {"sections": [
        _rendered_section(
            "cost_optimization", title="COST OPTIMIZATION", tier="conditional",
            coverage_content=["We use spot instances in non-production."],
        ),
    ]}
    rendered = build_rendered_sections(knowledge_object)
    assert len(rendered) == 1
