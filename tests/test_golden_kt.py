"""Golden KT test (Phase 21): runs a fixed transcript through the real,
full pipeline orchestration (pipeline.run_kt_pipeline — the same entry point
/kt-from-transcript uses) end to end: classification -> field population ->
knowledge object assembly -> rendering -> validation -> PDF export.

Deliberately LLM-free for determinism/speed/no API cost: monkeypatches
pipeline.get_llm_provider and ai.get_llm_provider to return None (patching
each importing module's own bound name from `from llm_provider import
get_llm_provider` — patching llm_provider.get_llm_provider itself wouldn't
affect either, since both already hold direct references). This exercises
the non-LLM-dependent majority of the pipeline (pattern/semantic field
population, knowledge object assembly, rendering, validation) as a standing
regression guard. LLM-dependent structured extraction (security_controls,
disaster_recovery, etc.) is out of scope here — already covered by this
session's documented live manual verification passes (REPOSITORY_AUDIT.md).

Also bypasses pipeline.load_models() (which would additionally load a
Whisper speech-to-text model this text-only test never uses) by constructing
MAPPER_PIPELINE directly.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import ai
import pipeline
from context_mapper import ContextMappingPipeline
from devops_transcription import clean_transcript
from kt_schema_loader import SCHEMA

TRANSCRIPT = """This is a knowledge transfer session for the Order Processing platform.
Business criticality is High, since this is one of the company's most business critical systems.
The system is used internally by the fulfillment and customer support teams to process orders.
If the system goes down, customers cannot place orders and warehouse fulfillment stops immediately.

All services run on Amazon EKS. Traffic enters through CloudFront before reaching the Kubernetes workload.
The primary database runs on Amazon RDS with PostgreSQL.

Never modify Terraform state files manually, as recovering from state corruption is extremely difficult.
Do not alter the production Kubernetes cluster autoscaler configuration, because incorrect changes can impact the entire platform.

The most common production issue is database connection pool exhaustion during flash sales.
Another common issue is missing Kubernetes secrets causing pods to crash-loop after deployment.

The platform becomes extremely busy during Black Friday and month-end sales.
Deployments must be avoided during those periods.

Developers own the application code while the platform engineering team owns the Kubernetes infrastructure.
The escalation chain starts with an on-call engineer followed by the platform engineering manager and then the head of engineering.
"""


@pytest.fixture
def golden_pipeline(monkeypatch):
    """Wires up a deterministic, LLM-free MAPPER_PIPELINE without loading Whisper."""
    monkeypatch.setattr(pipeline, "get_llm_provider", lambda: None)
    monkeypatch.setattr(ai, "get_llm_provider", lambda: None)
    pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
    yield pipeline
    pipeline.MAPPER_PIPELINE = None


def test_golden_kt_pipeline_end_to_end(golden_pipeline):
    cleaned = clean_transcript(TRANSCRIPT)
    result = golden_pipeline.run_kt_pipeline("golden-test-job", cleaned)

    assert result["status"] == "completed", result.get("error")

    coverage = result["coverage"]
    for section_id in ("system_overview", "danger_zones", "common_failures", "known_bad_days", "ownership_escalation"):
        status = coverage.get(section_id, {}).get("status")
        assert status in ("weak", "covered"), f"{section_id} expected weak/covered, got {status!r}"

    # Phase 19's validation layer as a standing regression guard — any
    # future schema/field-id mismatch this pipeline introduces should show
    # up here, not require someone to notice a blank PDF section by hand.
    assert result["validation_warnings"] == [], result["validation_warnings"]

    # Phase 22's quality score should be present and well-formed.
    quality = result["quality_score"]
    assert 0.0 <= quality["overall_score"] <= 100.0
    assert quality["grade"] in {"A", "B", "C", "D", "F"}

    # Proves specialized renderers are actually firing (not every section
    # silently falling back to generic narrative) — danger_zones should be a
    # WarningBlock given the section_rules.py force-classification rule for
    # "never modify"/"autoscaler configuration".
    rendered_sections = result["knowledge_object"]["rendered_sections"]
    assert rendered_sections
    block_types = {b["type"] for sec in rendered_sections for b in sec.get("blocks", [])}
    assert block_types - {"NarrativeBlock"}, f"every section fell back to narrative: {block_types}"

    danger_zones = next(sec for sec in rendered_sections if sec["section_id"] == "danger_zones")
    assert any(b["type"] == "WarningBlock" for b in danger_zones["blocks"])

    # Concrete pattern-derived facts, not just structural shape.
    danger_text = " ".join(
        w for sec in rendered_sections if sec["section_id"] == "danger_zones"
        for b in sec["blocks"] for w in b.get("warnings", [])
    ).lower()
    assert "terraform" in danger_text
    assert "autoscaler" in danger_text


def test_golden_kt_pipeline_produces_a_valid_pdf(golden_pipeline):
    from pdf_rendering import render_pdf_html
    from weasyprint import HTML

    cleaned = clean_transcript(TRANSCRIPT)
    result = golden_pipeline.run_kt_pipeline("golden-test-job-pdf", cleaned)
    assert result["status"] == "completed", result.get("error")

    knowledge_object = result["knowledge_object"]
    html_doc = render_pdf_html(
        title=knowledge_object.get("system_name") or "KT Document",
        job_id="golden-test-job-pdf",
        rendered_sections=knowledge_object["rendered_sections"],
        coverage=result["coverage"],
        date_str="05 September 2026",
    )
    pdf_bytes = HTML(string=html_doc, base_url=os.path.dirname(os.path.dirname(__file__))).write_pdf()
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert pdf_bytes.startswith(b"%PDF")
