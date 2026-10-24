from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import pytest
from pdf_rendering import _build_fallback_paragraphs, build_rendered_sections, render_pdf_html


def test_render_pdf_html_returns_html_string():
    sections = [
        {
            "section_id": "system_overview",
            "section_title": "System Overview",
            "blocks": [
                {
                    "type": "NarrativeBlock",
                    "title": "Summary",
                    "paragraphs": ["Platform operates across three regions."],
                }
            ],
        }
    ]

    html_doc = render_pdf_html(
        title="Test KT",
        job_id="job-123",
        rendered_sections=sections,
        coverage={},
        date_str="28 July 2026",
    )

    assert isinstance(html_doc, str)
    assert "Test KT" in html_doc
    assert "Platform operates across three regions." in html_doc


def test_weasyprint_generates_pdf_from_html():
    try:
        from weasyprint import HTML
    except Exception:
        pytest.skip("WeasyPrint native dependencies are not installed")

    sections = [
        {
            "section_id": "system_overview",
            "section_title": "System Overview",
            "blocks": [
                {
                    "type": "NarrativeBlock",
                    "title": "Summary",
                    "paragraphs": ["Platform operates across three regions."],
                }
            ],
        }
    ]

    html_doc = render_pdf_html(
        title="Test KT",
        job_id="job-123",
        rendered_sections=sections,
        coverage={},
        date_str="28 July 2026",
    )
    pdf_bytes = HTML(string=html_doc, base_url=os.path.dirname(os.path.dirname(__file__))).write_pdf()

    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert pdf_bytes.startswith(b"%PDF")


def test_build_fallback_paragraphs_dedupes_repeated_content():
    section = {
        "coverage_content": [
            "System overview content from coverage.",
            "System overview content from coverage.",
        ],
        "facts": [
            {"id": "system_name", "label": "System Name", "value": "System overview content from coverage."}
        ],
        "evidence": [
            {"text": "System overview content from coverage."}
        ],
        "description": "System overview content from coverage.",
    }

    paragraphs = _build_fallback_paragraphs(section)

    assert len(paragraphs) == 1
    assert paragraphs[0] == "System overview content from coverage."


# --------------------------------------------------------------------------
# Short KTs, shorter PDFs (13 Oct): a 112-word KT was 12 pages.
# --------------------------------------------------------------------------

from pdf_rendering import (  # noqa: E402
    NOT_COVERED_SECTION_ID, _is_represented, _norm_for_match, build_job_pdf, collapse_not_covered_sections,
)
from renderers.blocks.common import NOT_COVERED_MESSAGE, no_coverage_block  # noqa: E402
from renderers.sections.architecture_reference import CONNECTIONS_TABLE_TITLE  # noqa: E402


def _section(sid, title, blocks):
    return {"section_id": sid, "section_title": title, "blocks": blocks}


def _narrative(text):
    return {"type": "NarrativeBlock", "title": "Summary", "paragraphs": [text]}


def test_sections_with_nothing_from_the_kt_are_listed_once_before_the_digests():
    sections = [
        _section("system_overview", "System Overview", [_narrative("Paygate authorises card payments.")]),
        _section("day1_survival_checklist", "Day-1 Survival Checklist", [no_coverage_block("Day-1 Survival Checklist")]),
        _section("security_controls", "Security", [_narrative("Secrets are in Key Vault.")]),
        _section("known_bad_days", "Operational Calendar", [no_coverage_block("Operational Calendar")]),
        _section("kt_coverage", "KT Coverage", [_narrative("Coverage matrix.")]),
    ]
    out = collapse_not_covered_sections(sections)
    assert [s["section_id"] for s in out] == ["system_overview", "security_controls", NOT_COVERED_SECTION_ID, "kt_coverage"]
    listed = out[2]["blocks"][1]["items"]
    assert listed == ["Day-1 Survival Checklist", "Operational Calendar"]
    html_doc = render_pdf_html(title="T", job_id="job-1", rendered_sections=sections, coverage={}, date_str="d")
    assert NOT_COVERED_MESSAGE not in html_doc and "Not Covered in This KT" in html_doc


def test_the_not_covered_list_names_what_the_coverage_matrix_calls_not_covered():
    # A user's KT (23 Oct, matrix T03): the list named Sign-off (recorded at
    # review, "After review" in the matrix) and left out Handover Completion
    # Check (Missing in the matrix, omitted from the document when empty), so
    # it disagreed with the matrix and the cover's "not covered" count.
    from pdf_rendering import SIGNOFF_PENDING, missing_sections

    ko = {"sections": [{"id": "kt_coverage", "_coverage_rows": [
        {"Domain": "Security", "Coverage": "Missing", "Assessment": "Not covered in the KT session."},
        {"Domain": "Disaster Recovery", "Coverage": "Missing",
         "Assessment": "Discussed, but stated as not in place: “There is no DR plan.”"},
        {"Domain": "Handover Completion Check", "Coverage": "Missing", "Assessment": "Not covered in the KT session."},
        {"Domain": "Sign-off", "Coverage": "After review", "Assessment": "Recorded when the document is reviewed."},
    ]}]}
    assert missing_sections(ko) == ["Security", "Handover Completion Check"]
    sections = [
        _section("security_controls", "Security", [no_coverage_block("Security")]),
        _section("disaster_recovery", "Disaster Recovery", [_narrative("There is no DR plan.")]),
        _section("kt_coverage", "KT Coverage", [_narrative("Coverage matrix.")]),
        _section("signoff", "Sign-off", [no_coverage_block("Sign-off")]),
    ]
    out = collapse_not_covered_sections(sections, missing_sections(ko))
    assert [s["section_id"] for s in out] == ["disaster_recovery", NOT_COVERED_SECTION_ID, "kt_coverage", "signoff"]
    assert out[1]["blocks"][1]["items"] == ["Security", "Handover Completion Check"]
    assert out[3]["blocks"][0]["paragraphs"] == [SIGNOFF_PENDING]


def test_a_section_with_its_own_content_is_never_folded_into_the_list():
    sections = [_section("disaster_recovery", "Disaster Recovery",
                         [_narrative("RDS takes daily backups."), no_coverage_block("Disaster Recovery")])]
    assert collapse_not_covered_sections(sections) == sections


def test_the_connections_table_prints_the_sentence_number_not_the_sentence():
    table = {"type": "DecisionTable", "title": CONNECTIONS_TABLE_TITLE,
             "columns": ["From", "Connection", "To", "Said in the KT"],
             "rows": [{"From": "Dataflow", "Connection": "writes to", "To": "BigQuery",
                       "Said in the KT": "Dataflow jobs process them, and the results land in BigQuery."}],
             "sources": [[2]]}
    html_doc = render_pdf_html(title="T", job_id="job-1", rendered_sections=[
        _section("architecture_reference", "Architecture", [table])], coverage={}, date_str="d")
    assert 'sentence <a href="#src-2">2</a>' in html_doc
    assert "the results land in BigQuery" not in html_doc


def test_without_the_contents_page_the_cover_lists_the_sections():
    sections = [_section("system_overview", "System Overview", [_narrative("Paygate authorises card payments.")])]
    with_toc = render_pdf_html(title="T", job_id="job-1", rendered_sections=sections, coverage={}, date_str="d")
    without = render_pdf_html(title="T", job_id="job-1", rendered_sections=sections, coverage={}, date_str="d",
                              show_toc=False)
    assert 'class="toc-page"' in with_toc and 'class="cover-contents"' not in with_toc
    assert 'class="toc-page"' not in without and 'class="cover-contents"' in without


def test_a_short_document_has_no_contents_page():
    pytest.importorskip("weasyprint")
    from pypdf import PdfReader
    import io
    job = {"knowledge_object": {"system_name": "Paygate", "rendered_sections": [
        _section("system_overview", "System Overview", [_narrative("Paygate authorises card payments.")])]}}
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(build_job_pdf("job-1", job))).pages)
    assert "TABLE OF CONTENTS" not in text.upper() and "CONTENTS" in text.upper()
    assert "System Overview" in text.split("01 System Overview")[0]     # listed on the cover


def test_a_sentence_shown_clause_by_clause_as_fields_is_not_repeated():
    rendered = [_norm_for_match(s) for s in ["RTO (Recovery Time Objective) two hours",
                                             "RPO (Recovery Point Objective) 30 minutes"]]
    assert _is_represented("Our RTO is two hours and our RPO is 30 minutes.", rendered)
    assert not _is_represented("Our RTO is two hours and restores are tested monthly.", rendered)
    # A one-word list item is not a clause: the sentence is judged whole.
    assert not _is_represented("Prometheus and Grafana cover monitoring.", [_norm_for_match("Prometheus covers metrics")])


def test_the_stylesheet_defines_every_variable_it_uses():
    import re
    css = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "pdf", "templates", "kt_document.css"),
               encoding="utf-8").read()
    used = set(re.findall(r"var\((--[\w-]+)\)", css))
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    assert used <= defined, used - defined
