"""P0-5: corrections, superseded tools and contradictions. Before: the first
stated value always won (a withdrawn RTO was headlined, an abandoned paging
tool was named), and contradictory rules were printed side by side with no
flag."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from conflicts import find_contradictions, find_superseded_tools, resolve_single_value
from field_populator import extract_rto_rpo


@pytest.mark.parametrize("text,expected", [
    ("The RTO is four hours. The RTO is thirty minutes, the four hours was the old target.", "thirty minutes (replaces four hours"),
    ("The RTO is 8 hours. The RTO is 2 hours now; 8 hours was the old target.", "2 hours (replaces 8 hours"),
    ("RTO is two hours. Later on: RTO is one hour.", "Conflicting values stated: two hours; one hour"),
    ("The RTO is four hours. As we said, RTO is 4 hours.", "four hours"),
    ("The recovery time objective, RTO, is 2 hours and the recovery point objective, RPO, is 15 minutes.", "2 hours"),
])
def test_rto_follows_corrections_and_flags_conflicts(text, expected):
    assert extract_rto_rpo(text)["rto_metric"].startswith(expected)


def test_correction_markers_pick_the_latest_corrected_value():
    value, note = resolve_single_value([("four hours", "The RTO is four hours."),
                                        ("thirty minutes", "Currently the RTO is thirty minutes.")])
    assert value == "thirty minutes" and "four hours" in note


def test_superseded_tools_and_their_replacements():
    found = find_superseded_tools([
        "Alerts go to Opsgenie now, we moved off PagerDuty last month.",
        "We migrated from Jenkins to GitHub Actions in March.",
        "We replaced Splunk with Datadog.",
        "PagerDuty pages us when checkout fails.",          # no change described
    ])
    assert {k: v["new"] for k, v in found.items()} == {"pagerduty": "Opsgenie", "jenkins": "GitHub Actions",
                                                      "splunk": "Datadog"}


def test_contradictory_rules_are_paired_and_unrelated_rules_are_not():
    pairs = find_contradictions([
        "Deploying on Fridays is fine for us.",
        "Never deploy on Fridays, we had an outage last time.",
        "Never delete items from the bookings table by hand.",
        "Restarting the consumer is safe.",
    ])
    assert pairs == [("Deploying on Fridays is fine for us.", "Never deploy on Fridays, we had an outage last time.")]


@pytest.mark.parametrize("text,team", [
    ("The payments team owns the service. Actually the platform team owns it since the reorg.",
     "Platform team (previously Payments team)"),
    ("The payments team owns the service. The platform team owns the service.",
     "Conflict: Payments team or Platform team"),
])
def test_two_owners_for_one_thing_are_resolved_or_flagged(text, team):
    from renderers.sections.ownership_escalation import render

    out = render({"id": "ownership_escalation", "title": "Ownership", "fields": {}, "coverage_content": [text]})
    rows = [r for b in out["blocks"] if b["type"] == "OwnershipTable" for r in b["rows"]]
    assert len(rows) == 1 and rows[0]["team"].startswith(team)


CONTRADICTORY = """This KT is for Orbitpay, our payout service on AWS Lambda with DynamoDB.
The RTO is four hours. Actually the RTO is thirty minutes, the four hours was the old target.
Deploying on Fridays is fine for us. Never deploy on Fridays, we had an outage last time.
Alerts go to PagerDuty. Alerts go to Opsgenie now, we moved off PagerDuty last month.
The payments team owns the service. Actually the platform team owns it since the reorg.
"""


@pytest.fixture(scope="module")
def contradictory_result():
    import ai
    import pipeline
    from context_mapper import ContextMappingPipeline
    from devops_transcription import clean_transcript
    from kt_schema_loader import SCHEMA

    saved = (pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE)
    pipeline.get_llm_provider = ai.get_llm_provider = lambda: None
    pipeline.record_candidates = lambda *a, **k: None
    pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
    try:
        yield pipeline.run_kt_pipeline("conflicts-0001", clean_transcript(CONTRADICTORY))
    finally:
        pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE = saved


def _section(result, sid):
    return next(s for s in result["knowledge_object"]["rendered_sections"] if s["section_id"] == sid)


def test_document_uses_the_corrected_values(contradictory_result):
    dr = json.dumps(_section(contradictory_result, "disaster_recovery"))
    assert "RTO (Recovery Time Objective): thirty minutes" in dr
    owner_rows = [r for b in _section(contradictory_result, "ownership_escalation")["blocks"]
                  if b["type"] == "OwnershipTable" for r in b["rows"]]
    assert {"role": "On-call tool", "team": "Opsgenie"} in owner_rows
    assert any(r["team"].startswith("Platform team (previously Payments team)") for r in owner_rows)
    tech = json.dumps(_section(contradictory_result, "system_overview"))
    assert "Opsgenie" in tech and "PagerDuty" not in tech


def test_document_flags_the_contradictory_rule(contradictory_result):
    warnings = [w for s in contradictory_result["knowledge_object"]["rendered_sections"] for b in s["blocks"]
                if b["type"] == "WarningBlock" and "conflict" in (b.get("title") or "").lower() for w in b["warnings"]]
    assert any("Fridays is fine" in w and "Never deploy on Fridays" in w for w in warnings)
    gaps = next(s for s in contradictory_result["knowledge_object"]["sections"] if s["id"] == "kt_coverage")["_knowledge_gaps"]
    assert any(g.startswith("Possible conflict") for g in gaps)
