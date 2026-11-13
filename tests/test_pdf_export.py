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
    CONNECTIONS_APPENDIX_TITLE, _is_represented, _norm_for_match, build_job_pdf, collapse_not_covered_sections,
    not_covered_titles,
)
from renderers.blocks.common import NOT_COVERED_MESSAGE, no_coverage_block  # noqa: E402
from renderers.sections.architecture_reference import CONNECTIONS_TABLE_TITLE  # noqa: E402


def _section(sid, title, blocks):
    return {"section_id": sid, "section_title": title, "blocks": blocks}


def _narrative(text):
    return {"type": "NarrativeBlock", "title": "Summary", "paragraphs": [text]}


def test_sections_with_nothing_from_the_kt_are_left_out_of_the_body():
    # The PDF clean-up (12 Nov): they were listed again in a "Not Covered in
    # This KT" section, beside the coverage matrix's Missing rows, the
    # cover's "not covered" tile and the follow-up list.
    sections = [
        _section("system_overview", "System Overview", [_narrative("Paygate authorises card payments.")]),
        _section("day1_survival_checklist", "Day-1 Survival Checklist", [no_coverage_block("Day-1 Survival Checklist")]),
        _section("security_controls", "Security", [_narrative("Secrets are in Key Vault.")]),
        _section("known_bad_days", "Operational Calendar", [no_coverage_block("Operational Calendar")]),
        _section("kt_coverage", "KT Coverage", [_narrative("Coverage matrix.")]),
    ]
    out = collapse_not_covered_sections(sections)
    assert [s["section_id"] for s in out] == ["system_overview", "security_controls", "kt_coverage"]
    html_doc = render_pdf_html(title="T", job_id="job-1", rendered_sections=sections, coverage={}, date_str="d")
    assert NOT_COVERED_MESSAGE not in html_doc and "Not Covered in This KT" not in html_doc


def test_what_is_left_out_is_what_the_coverage_matrix_calls_not_covered():
    # A user's KT (23 Oct, matrix T03): the list named Sign-off (recorded at
    # review, "After review" in the matrix) and left out Handover Completion
    # Check (Missing in the matrix, omitted from the document when empty), so
    # it disagreed with the matrix and the cover's "not covered" count.
    from pdf_rendering import missing_sections

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
    # A sign-off not started is left out too: its section shows the record
    # once one starts (signoff.apply_to_document).
    assert [s["section_id"] for s in out] == ["disaster_recovery", "kt_coverage"]
    assert not_covered_titles(sections, ko) == ["Security", "Handover Completion Check"]
    # A section the matrix calls not covered but shown with content is not
    # counted as left out, so the matrix check (gaps_consistent) catches it.
    shown = sections + [_section("handover_completion_check", "Handover Completion Check", [_narrative("Done.")])]
    assert not_covered_titles(shown, ko) == ["Security"]


def test_a_section_the_matrix_calls_not_covered_shows_no_borrowed_field_value():
    # A user's KT (U05): Deployment & Rollback, not discussed, showed
    # "Terraform Workspace / State Location: The Data Engineering team owns
    # database restoration, ..." (a sentence shown in full under Ownership),
    # while the coverage matrix and the cover called it not covered.
    borrowed = "The Data Engineering team owns database restoration, while the Platform team recreates infrastructure."
    ko = {"_not_covered_sections": ["deployment_and_rollback"], "sections": [
        {"id": "deployment_and_rollback", "title": "Deployment & Rollback", "coverage_content": [],
         "facts": [{"id": "terraform_state", "label": "Terraform Workspace / State Location", "value": borrowed}]},
        {"id": "ownership_escalation", "title": "Ownership & Escalation", "coverage_content": [borrowed]}]}
    rendered = build_rendered_sections(ko)
    deployment = next(s for s in rendered if s["section_id"] == "deployment_and_rollback")
    assert [p for b in deployment["blocks"] for p in b.get("paragraphs") or []] == [NOT_COVERED_MESSAGE]
    assert collapse_not_covered_sections(rendered, ["Deployment & Rollback"])[0]["section_id"] == "ownership_escalation"


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
        _section("architecture_reference", "Architecture", [_narrative("Dataflow writes to BigQuery."), table]),
        _section("monitoring", "Monitoring", [_narrative("Grafana shows the job lag.")])], coverage={}, date_str="d")
    assert 'sentence <a href="#src-2">2</a>' in html_doc
    assert "the results land in BigQuery" not in html_doc
    # Evidence for the diagram, listed at the end (the PDF clean-up, 12 Nov):
    # after the last section, not between the diagram and the next section.
    assert html_doc.index("Grafana shows the job lag") < html_doc.index('id="diagram-connections"') \
        < html_doc.index("sentence <a")
    assert CONNECTIONS_APPENDIX_TITLE in html_doc.split('id="diagram-connections"')[1]


def test_without_the_contents_page_the_cover_lists_the_sections():
    sections = [_section("system_overview", "System Overview", [_narrative("Paygate authorises card payments.")])]
    with_toc = render_pdf_html(title="T", job_id="job-1", rendered_sections=sections, coverage={}, date_str="d")
    without = render_pdf_html(title="T", job_id="job-1", rendered_sections=sections, coverage={}, date_str="d",
                              show_toc=False)
    assert 'class="toc-page"' in with_toc and 'class="cover-contents"' not in with_toc
    assert 'class="toc-page"' not in without and 'class="cover-contents"' in without


def test_the_cover_lists_the_sections_whenever_that_saves_the_contents_page():
    pytest.importorskip("weasyprint")
    from pypdf import PdfReader
    import io
    # Sixteen sections of a few lines each: twelve pages or so, which had a
    # contents page of its own (a page for a dozen lines) above nine pages.
    sections = [_section(f"s{i}", f"Section number {i}", [_narrative(" ".join(["Paygate authorises card payments "
                "through Adyen and settles them nightly into the ledger."] * 9))] * 3) for i in range(16)]
    job = {"knowledge_object": {"system_name": "Paygate", "rendered_sections": sections}}
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(build_job_pdf("job-1", job))).pages)
    assert "TABLE OF CONTENTS" not in text.upper() and "Section number 15" in text.split("01 Section number 0")[0]


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


def test_a_long_technology_list_is_two_pairs_a_row_and_a_short_kt_can_share_its_cover():
    grid = {"type": "TechnologyGrid", "title": "Technology summary",
            "rows": [{"label": f"Layer {i}", "value": f"Tool {i}"} for i in range(9)]}
    html_doc = render_pdf_html(title="T", job_id="job-1", rendered_sections=[
        _section("system_overview", "System Overview", [grid])], coverage={}, date_str="d", show_toc=False,
        compact_cover=True)
    assert 'class="kv-table kv-two"' in html_doc and "Tool 8" in html_doc
    assert 'class="cover-page cover-compact"' in html_doc
    # Not with a contents page.
    assert 'class="cover-page cover-compact"' not in render_pdf_html(title="T", job_id="job-1", rendered_sections=[
        _section("system_overview", "System Overview", [grid])], coverage={}, date_str="d", compact_cover=True)


def test_a_sentence_as_said_is_not_taken_for_another_ones_paraphrase():
    # Dev d09: "Escalations should be directed to Laura Bennett, Housing
    # Systems Manager." (shown in a row) shares three of four words with "The
    # system is owned by the housing systems team.", which was left out and
    # put back by the completeness check under Additional Notes.
    from pdf_rendering import RESIDUAL_BLOCK_TITLE, append_residual_content
    said = ["The system is owned by the housing systems team.",
            "Escalations should be directed to Laura Bennett, Housing Systems Manager."]
    section = {"id": "ownership_escalation", "coverage_content": list(said), "_source_sentences": list(said)}
    rendered = {"section_id": "ownership_escalation", "blocks": [{"type": "OwnershipTable", "title": "Ownership",
                "rows": [{"role": "Incident escalation", "team": said[1]}]}]}
    out = append_residual_content(section, rendered)
    residual = next(b for b in out["blocks"] if b["title"] == RESIDUAL_BLOCK_TITLE)
    assert residual["paragraphs"] == [said[0]]


def test_field_values_the_sentences_below_hold_word_for_word_are_not_shown_twice():
    from pdf_rendering import _repeats_said, _norm_for_match
    said = [_norm_for_match(s) for s in [
        "For monitoring, we use Prometheus and Grafana.",
        "CloudWatch collects AWS infrastructure logs, and PagerDuty sends incident notifications.",
        "Disaster recovery testing happens twice annually, but the current RTO and RPO need confirmation.",
        "We can restore everything within 4 hours."]]
    tools = {"type": "TechnologyGrid", "title": "Monitoring Stack",
             "rows": [{"label": "Tools", "value": "Prometheus, Grafana, CloudWatch, PagerDuty"}]}
    routing = {"type": "NarrativeBlock", "title": "Alert Routing", "paragraphs": ["PagerDuty sends incident notifications"]}
    frequency = {"type": "NarrativeBlock", "title": "", "paragraphs": ["DR testing frequency: twice annually"]}
    assert _repeats_said(tools, said) and _repeats_said(routing, said) and _repeats_said(frequency, said)
    # A short value whose label the sentence does not name stays: it says what the value is.
    assert not _repeats_said({"type": "NarrativeBlock", "title": "", "paragraphs": ["RTO: 4 hours"]}, said)
    # The inventory gathered across the KT stays, and so does a value the sentences do not hold.
    assert not _repeats_said(dict(tools, title="Technology summary"), said)
    assert not _repeats_said({"type": "TechnologyGrid", "title": "Monitoring Stack",
                              "rows": [{"label": "Tools", "value": "Prometheus, Datadog"}]}, said)


def test_the_architecture_sentences_can_go_before_the_diagram():
    from pdf_rendering import details_before_diagram
    diagram = {"type": "DiagramBlock", "title": "High-Level Architecture", "svg_uri": "data:,"}
    details = {"type": "NarrativeBlock", "title": "Architecture Details", "paragraphs": ["Redis caches sessions."]}
    sections = [_section("architecture_reference", "Architecture", [diagram, details]),
                _section("monitoring", "Monitoring", [_narrative("Grafana shows the job lag.")])]
    moved = details_before_diagram(sections)
    assert [b["title"] for b in moved[0]["blocks"]] == ["Architecture Details", "High-Level Architecture"]
    assert [b["title"] for b in sections[0]["blocks"]] == ["High-Level Architecture", "Architecture Details"]  # a copy
    assert moved[1] is sections[1]
