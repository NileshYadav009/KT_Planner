"""P0-6: a KT built with failed stages or failed LLM calls must say so.

Before: every stage failure was logged and swallowed, so a job reported
"completed" while the document was missing its tables, or every section, or
was built entirely without the LLM, and nothing told the reader."""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai
import pipeline
from context_mapper import ContextMappingPipeline
from devops_transcription import clean_transcript
from kt_schema_loader import SCHEMA
from llm.cached_provider import CachedLLMProvider
from pdf_rendering import render_pdf_html

TRANSCRIPT = clean_transcript(
    "This KT is for TripWise, the booking backend behind our travel app. "
    "Bookings are written to DynamoDB and payments go through Stripe. "
    "Datadog is our monitoring tool and PagerDuty pages us if booking success drops below ninety five percent. "
    "RTO is four hours. "
    "Never delete items from the bookings table by hand. "
    "Escalate to Marco Silva if PagerDuty is not acknowledged within fifteen minutes."
)


@pytest.fixture(scope="module")
def mapper():
    return ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)


@pytest.fixture
def no_llm(monkeypatch, mapper):
    monkeypatch.setattr(pipeline, "get_llm_provider", lambda: None)
    monkeypatch.setattr(ai, "get_llm_provider", lambda: None)
    monkeypatch.setattr(pipeline, "record_candidates", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "MAPPER_PIPELINE", mapper)
    return pipeline


def _boom(*args, **kwargs):
    raise RuntimeError("simulated crash")


def test_clean_run_is_completed_and_says_no_llm_was_used(no_llm):
    result = no_llm.run_kt_pipeline("status-ok-0001", TRANSCRIPT)
    assert result["status"] == "completed"
    assert result["warnings"] == []
    assert any("without an LLM" in n for n in result["notices"])


def test_field_population_crash_is_disclosed(no_llm, monkeypatch):
    monkeypatch.setattr(pipeline, "populate_fields", _boom)
    result = no_llm.run_kt_pipeline("status-fields-0001", TRANSCRIPT)
    assert result["status"] == "completed_with_warnings"
    assert any("Field extraction failed" in w for w in result["warnings"])
    assert any(e["stage"] == "Field population" for e in result["stage_errors"])


@pytest.mark.parametrize("stage_fn", ["build_knowledge_object", "build_rendered_sections"])
def test_crash_without_which_there_is_no_document_fails_the_job(no_llm, monkeypatch, stage_fn):
    monkeypatch.setattr(pipeline, stage_fn, _boom)
    result = no_llm.run_kt_pipeline(f"status-{stage_fn[:8]}-0001", TRANSCRIPT)
    assert result["status"] == "failed"
    assert "failed" in result["error"]


def test_failed_llm_calls_are_disclosed(no_llm, monkeypatch):
    monkeypatch.setenv("LLM_CACHE", "off")

    class Down:
        model = "fake"

        def generate(self, prompt, *args, **kwargs):
            raise RuntimeError("Error code: 429 - rate limit")

    provider = CachedLLMProvider(Down())
    monkeypatch.setattr(pipeline, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(ai, "get_llm_provider", lambda: provider)
    result = no_llm.run_kt_pipeline("status-llm-0001", TRANSCRIPT)
    assert result["status"] == "completed_with_warnings"
    assert result["llm_usage"]["failed"] > 0
    assert any("LLM calls failed" in w for w in result["warnings"])


def test_pdf_cover_shows_warnings_escaped():
    html = render_pdf_html(title="T", job_id="job-1234", rendered_sections=[], coverage={}, date_str="d",
                           warnings=["Field extraction failed <b>x</b>"], notices=["Generated without an LLM."])
    assert "Review before relying on this document" in html
    assert "Field extraction failed &lt;b&gt;x&lt;/b&gt;" in html
    assert "Generated without an LLM." in html


def test_pdf_has_no_banner_when_nothing_failed():
    html = render_pdf_html(title="T", job_id="job-1234", rendered_sections=[], coverage={}, date_str="d")
    assert "Review before relying" not in html


def test_export_route_accepts_and_prints_warnings():
    pypdf = pytest.importorskip("pypdf")
    from fastapi.testclient import TestClient
    import main

    job_id = "status-route-0001"
    pipeline.JOB_QUEUE[job_id] = {
        "status": "completed_with_warnings", "coverage": {},
        "warnings": ["3 of 10 LLM calls failed (provider error or quota)."],
        "knowledge_object": {"system_name": "T", "rendered_sections": [
            {"section_id": "s", "section_title": "S", "blocks": [{"type": "NarrativeBlock", "title": "S", "paragraphs": ["x"]}]}]},
    }
    try:
        resp = TestClient(main.app).get(f"/export/pdf/{job_id}")
    finally:
        pipeline.JOB_QUEUE.pop(job_id, None)
    assert resp.status_code == 200
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(resp.content)).pages)
    assert "Review before relying" in text and "3 of 10 LLM calls failed" in text
