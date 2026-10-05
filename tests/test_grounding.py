"""P0-4: LLM output reaches the KT document only when the transcript supports
it. Before: a fake model's invented contact, region, SLA, tools and CLI steps
all reached the PDF unmarked, and malformed replies were printed as values."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from grounding import added_specifics, ground_structured, looks_malformed, ungrounded

SOURCE = ("Datadog is our monitoring tool. The dashboard to watch is booking success rate, and PagerDuty "
          "pages us if it drops below ninety five percent. Escalate to Marco Silva if PagerDuty is not "
          "acknowledged within fifteen minutes. Around two hundred thousand travellers use it each month.")


@pytest.mark.parametrize("value", [
    "Check the booking success rate dashboard in Datadog",
    "Verify if the success rate has dropped below 95%",
    "Escalate to Marco Silva after 15 minutes without acknowledgement",
    "About 200,000 travellers per month",
    "PagerDuty",
    "High",
])
def test_paraphrases_of_stated_facts_are_grounded(value):
    assert ungrounded(value, SOURCE) == []


@pytest.mark.parametrize("value,expected", [
    ("Managed by John Smith in us-east-1 with a 99.99% availability SLA", {"John", "Smith", "us-east-1", "99.99"}),
    ("PagerDuty pages the #sre-war-room Slack channel", {"#sre-war-room", "Slack"}),
    ("Tools: Datadog, Splunk", {"Splunk"}),
    ("Restart the API with kubectl rollout restart", {"kubectl"}),
    ("RPO is five minutes", {"5"}),
])
def test_invented_specifics_are_caught(value, expected):
    assert expected <= set(ungrounded(value, SOURCE))


@pytest.mark.parametrize("value,bad", [
    ('Sure! Here is what I found: {"tools": ["Datadog", ', True),
    ("Here's the value: PagerDuty", True),
    ('{"oncall_tool": "PagerDuty"}', True),
    ("PagerDuty", False),
    ("Escalate to Marco Silva", False),
])
def test_malformed_replies_are_recognised(value, bad):
    assert looks_malformed(value) is bad


def test_structured_result_keeps_real_items_and_drops_invented_ones():
    data = {"tools": ["Datadog", "PagerDuty", "Splunk"],
            "first_response_steps": ["Check the booking success rate dashboard in Datadog",
                                     "Restart the booking API with kubectl rollout restart"],
            "alert_routing": "PagerDuty pages the #sre-war-room Slack channel"}
    dropped = []
    out = ground_structured(data, SOURCE, dropped)
    assert out["tools"] == ["Datadog", "PagerDuty"]
    assert out["first_response_steps"] == ["Check the booking success rate dashboard in Datadog"]
    assert out["alert_routing"] is None
    assert len(dropped) == 3


def test_polish_may_use_its_own_template_labels_but_not_new_facts():
    prompt = "Format as **Trigger:**, **Window:**, **Approver:** lines.\nSource fragments:\n- Deploys run every Tuesday evening."
    ok = "**Trigger:** Tuesday evening\n**Approver:** Not specified"
    invented = "**Trigger:** Tuesday evening, approved by Jane Doe in the #release channel"
    assert added_specifics(ok, ["Deploys run every Tuesday evening.", prompt]) == []
    assert {"Jane", "Doe", "#release"} <= set(added_specifics(invented, ["Deploys run every Tuesday evening.", prompt]))


# --------------------------------------------------------------------------
# End to end: a model that invents facts, through the real pipeline
# --------------------------------------------------------------------------

FABRICATED = ["John Smith", "us-east-1", "99.99", "Splunk", "sre-war-room", "kubectl", "VP Engineering"]


class Liar:
    model = "liar"

    def generate(self, prompt, *args, **kwargs):
        sp = kwargs.get("system_prompt") or ""
        if "JSON extractor" in sp:
            if '"oncall_tool"' in prompt:
                return json.dumps({"oncall_tool": "PagerDuty",
                                   "escalation_chain": "On-call engineer -> John Smith (VP Engineering) -> CTO",
                                   "application_ownership": None, "infrastructure_ownership": None,
                                   "operational_escalation_guidance": None})
            if '"first_response_steps"' in prompt:
                return json.dumps({"tools": ["Datadog", "PagerDuty", "Splunk"],
                                   "first_response_steps": ["Restart the booking API with kubectl rollout restart"],
                                   "alert_routing": "PagerDuty pages the #sre-war-room Slack channel"})
            return "{}"
        if prompt.startswith("You are filling a Knowledge Transfer document field."):
            return "Managed by John Smith in us-east-1 with a 99.99% availability SLA\nEXPLICIT"
        frags = [ln[2:] for ln in prompt.splitlines() if ln.startswith("- ")]
        if "technical writer" in sp and frags:
            return "\n".join(frags) + "\nThe service runs in us-east-1 with a 99.99% availability SLA."
        raise RuntimeError("unsupported prompt")


class Garbage:
    model = "garbage"

    def generate(self, prompt, *args, **kwargs):
        return 'Sure! Here is what I found: {"tools": ["Datadog", '


TRANSCRIPT = (
    "Today I'm handing over TripWise, the booking backend behind our travel mobile app. "
    "The app talks to an API on Amazon ECS with Fargate. Bookings are written to DynamoDB. "
    "Datadog is our monitoring tool. The dashboard to watch is booking success rate, and PagerDuty pages us "
    "if it drops below ninety five percent. "
    "One recurring problem is DynamoDB throttling during flash sales; switch the table to on-demand capacity. "
    "Never delete items from the bookings table by hand. RTO is four hours. "
    "Escalate to Marco Silva if PagerDuty is not acknowledged within fifteen minutes."
)


@pytest.fixture(scope="module")
def mapper():
    from context_mapper import ContextMappingPipeline
    from kt_schema_loader import SCHEMA
    return ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)


def _run_with(provider, monkeypatch, mapper, job_id):
    import ai
    import pipeline
    from devops_transcription import clean_transcript
    from llm.cached_provider import CachedLLMProvider

    monkeypatch.setenv("LLM_CACHE", "off")
    wrapped = CachedLLMProvider(provider)
    monkeypatch.setattr(pipeline, "get_llm_provider", lambda: wrapped)
    monkeypatch.setattr(ai, "get_llm_provider", lambda: wrapped)
    monkeypatch.setattr(pipeline, "record_candidates", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "MAPPER_PIPELINE", mapper)
    result = pipeline.run_kt_pipeline(job_id, clean_transcript(TRANSCRIPT))
    rendered = json.dumps(result["knowledge_object"]["rendered_sections"])
    return result, rendered


def test_invented_facts_never_reach_the_document(monkeypatch, mapper):
    result, rendered = _run_with(Liar(), monkeypatch, mapper, "ground-liar-0001")
    assert result["status"] in ("completed", "completed_with_warnings")
    leaked = [f for f in FABRICATED if f in rendered]
    assert leaked == [], leaked
    assert result["llm_usage"]["rejected"] > 0
    assert any("discarded" in n for n in result["notices"])
    # The real facts are still there.
    assert "Marco Silva" in rendered and "Datadog" in rendered


def test_malformed_replies_never_reach_the_document(monkeypatch, mapper):
    _, rendered = _run_with(Garbage(), monkeypatch, mapper, "ground-garbage-0001")
    assert "Sure! Here is what I found" not in rendered
