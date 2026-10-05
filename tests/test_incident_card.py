"""P1-8: the Quick Reference is an incident card. Before, TripWise's card had
3 rows and left out both known failures, the rollback and the danger zone:
it only looked for a failure mentioning "pod" or "deploy" and a danger zone
mentioning Terraform. Now: alert -> first check, each known failure -> fix,
rollback, who to page, never-do list, change freeze; every line is the
speaker's own words or a field value, and names its section."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from knowledge.knowledge_builder import append_quick_reference_section

TRIPWISE = """Alright, welcome. Today I'm handing over TripWise, the booking backend behind our travel mobile app. Around two hundred thousand travellers use it each month, mostly in Europe.

The app talks to an API on Amazon ECS with Fargate. Bookings are written to DynamoDB, payments go through Stripe, and confirmation emails are sent with SendGrid. Infrastructure is defined in CloudFormation.

We have three environments: dev, staging, and production. Staging uses Stripe test keys, so you can book freely there.

GitHub Actions builds the container and deploys to staging automatically on merge. Production needs a manual approval from the tech lead. If a release misbehaves, redeploy the previous task definition; that takes around ten minutes.

Datadog is our monitoring tool. The dashboard to watch is booking success rate, and PagerDuty pages us if it drops below ninety five percent.

One recurring problem is DynamoDB throttling during flash sales. The symptom is slow checkout, and the fix is to switch the table to on-demand capacity. Another issue is expired Stripe webhooks secrets, which silently stop payment confirmations; rotate the secret in Secrets Manager.

Never delete items from the bookings table by hand, since refunds depend on that history. And avoid deploying on Fridays or during the summer sale in July.

To save money we scale staging down to zero every night.

Backups use DynamoDB point in time recovery. RTO is four hours.

The mobile team owns the app, and our squad owns the backend. Escalate to Marco Silva if PagerDuty is not acknowledged within fifteen minutes.

In your first week, shadow the on-call and read the runbooks. By week three you should own releases.

The payment retry service migration is not yet completed, and nobody has picked it up.
"""


@pytest.fixture(scope="module")
def tripwise_card():
    import ai
    import pipeline
    from context_mapper import ContextMappingPipeline
    from devops_transcription import clean_transcript
    from kt_schema_loader import SCHEMA

    saved = (pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE)
    pipeline.get_llm_provider = ai.get_llm_provider = lambda: None          # the card must work without an LLM
    pipeline.record_candidates = lambda *a, **k: None
    pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
    try:
        result = pipeline.run_kt_pipeline("incident-card-0001", clean_transcript(TRIPWISE))
    finally:
        pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE = saved
        pipeline.JOB_STORE.delete("incident-card-0001")
    ko = result["knowledge_object"]
    rows = next(s for s in ko["sections"] if s["id"] == "quick_reference")["_quick_reference_rows"]
    rendered = next(s for s in ko["rendered_sections"] if s["section_id"] == "quick_reference")
    return rows, rendered


def _norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def test_tripwise_card_carries_every_annotated_incident_fact(tripwise_card):
    rows, _ = tripwise_card
    card = _norm(" ".join(f"{r['Situation']} {r['What to do']}" for r in rows))
    annotated = {
        "alert -> first check": "pagerduty pages us if it drops below ninety five percent",
        "failure 1": "dynamodb throttling during flash sales",
        "failure 1 fix": "switch the table to on demand capacity",
        "failure 2": "expired stripe webhooks secrets",
        "failure 2 fix": "rotate the secret in secrets manager",
        "rollback": "redeploy the previous task definition",
        "who to page": "escalate to marco silva",
        "never do": "never delete items from the bookings table by hand",
        "change freeze": "avoid deploying on fridays or during the summer sale in july",
        "recovery objective": "rto four hours",
    }
    missing = [name for name, phrase in annotated.items() if phrase not in card]
    assert missing == []


def test_every_line_is_sourced_from_the_kt(tripwise_card):
    rows, _ = tripwise_card
    transcript = _norm(TRIPWISE)
    for row in rows:
        assert row["Source"], row
        text = row["What to do"].replace("Recovery objectives: RTO ", "RTO is ")
        assert _norm(text) in transcript, row                    # the speaker's own words, nothing added


def test_the_card_renders_with_a_source_column(tripwise_card):
    _, rendered = tripwise_card
    table = rendered["blocks"][0]
    assert table["type"] == "DecisionTable"
    assert table["columns"] == ["Situation", "What to do", "Source"]


# --------------------------------------------------------------------------
# Building the card from sections
# --------------------------------------------------------------------------

def _section(sid, title, content=(), fields=None, structured=None):
    return {"id": sid, "title": title, "coverage_content": list(content), "fields": fields or {},
            "_structured": structured}


def _card(*sections, **extra):
    ko = {"sections": list(sections), "summary": {}}
    ko.update(extra)
    result = append_quick_reference_section(ko)
    qr = next((s for s in result["sections"] if s["id"] == "quick_reference"), None)
    return qr["_quick_reference_rows"] if qr else []


def test_failures_are_split_one_row_each_with_their_fix():
    rows = _card(_section("common_failures", "Common Failures", [
        "The most common failure is the Twilio rate limit.",
        "The fix is to scale the consumer down to two replicas.",
        "The second issue is MongoDB disk filling up during month end reporting, so we archive old results.",
        "There was a major outage last year caused by Redis memory saturation.",
    ]))
    assert [(r["Situation"], r["What to do"]) for r in rows] == [
        ("The Twilio rate limit", "The fix is to scale the consumer down to two replicas."),
        ("MongoDB disk filling up during month end reporting", "We archive old results."),
        ("Major outage last year caused by Redis memory saturation", "No fix was stated in the KT. Ask the outgoing owner."),
    ]
    assert {r["Source"] for r in rows} == {"Common Failures"}


def test_the_llms_structured_failures_are_used_when_present():
    rows = _card(_section("common_failures", "Common Failures", ["Something else entirely."],
                          structured={"failures": [{"symptom": "Checkout slow", "fix": "Switch to on-demand"}]}))
    assert rows == [{"Situation": "Checkout slow", "What to do": "Switch to on-demand", "Source": "Common Failures"}]


def test_a_runbook_step_filed_under_another_section_still_reaches_the_card():
    rows = _card(_section("security", "Security", [
        "When invoices get stuck, check whether the PDF worker pods are in CrashLoopBackOff.",
        "Renew it in Key Vault and restart the worker.",
        "Never rotate the SQL admin password without telling the finance on-call first.",
    ]))
    assert rows[0] == {"Situation": "Invoices get stuck",
                       "What to do": "Check whether the PDF worker pods are in CrashLoopBackOff. "
                                     "Renew it in Key Vault and restart the worker.",
                       "Source": "Security"}
    assert rows[1]["Situation"] == "Never" and rows[1]["Source"] == "Security"


def test_questions_gaps_and_plain_facts_are_not_guidance():
    rows = _card(
        _section("danger_zones", "Danger Zones", [
            "Who do I escalate to? I am not sure anymore, the old manager left.",
            "Service accounts use least privilege.",                     # a fact, not a caution
            "Be careful with the autoscaler, incorrect changes impact the whole platform.",
        ]),
        _section("ownership_escalation", "Ownership", ["Is there an escalation path?"]),
    )
    assert [(r["Situation"], r["What to do"]) for r in rows] == [
        ("Be careful", "Be careful with the autoscaler, incorrect changes impact the whole platform.")]


def test_corrected_and_contradicted_statements_are_not_given_as_orders():
    rows = _card(
        _section("monitoring_observability", "Monitoring", [
            "Alerts go to PagerDuty.", "Alerts go to Opsgenie now, we moved off PagerDuty last month."]),
        _section("known_bad_days", "Operational Calendar", [
            "Deploying on Fridays is fine for us.", "Never deploy on Fridays, we had an outage last time."]),
        _superseded_tools=[{"old": "PagerDuty", "new": "Opsgenie",
                            "quote": "Alerts go to Opsgenie now, we moved off PagerDuty last month."}],
        _conflicts=[{"section_id": "known_bad_days", "a": "Deploying on Fridays is fine for us.",
                     "b": "Never deploy on Fridays, we had an outage last time."}],
    )
    by_situation = {r["Situation"]: r["What to do"] for r in rows}
    assert by_situation["An alert fires"] == "Alerts go to Opsgenie now, we moved off PagerDuty last month."
    assert "Never" not in by_situation and "Do not deploy" not in by_situation
    conflict = by_situation["Conflicting guidance — confirm first"]
    assert "fine for us" in conflict and "Never deploy on Fridays" in conflict and "Ask the outgoing owner" in conflict


def test_spoken_lead_ins_are_dropped_and_duplicates_shown_once():
    rows = _card(
        _section("danger_zones", "Danger Zones", ["And never purge the dead letter queue."]),
        _section("security", "Security", ["Never purge the dead letter queue."]),
    )
    assert rows == [{"Situation": "Never", "What to do": "Never purge the dead letter queue.", "Source": "Danger Zones"}]
