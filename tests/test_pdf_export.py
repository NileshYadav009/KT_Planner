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
