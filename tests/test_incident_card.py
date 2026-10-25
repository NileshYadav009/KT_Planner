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
    return rows, rendered, [s["section_id"] for s in ko["rendered_sections"]]


def _norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def test_tripwise_card_carries_every_annotated_incident_fact(tripwise_card):
    rows, _, _ = tripwise_card
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
    rows, _, _ = tripwise_card
    transcript = _norm(TRIPWISE)
    for row in rows:
        assert row["Source"], row
        text = row["What to do"].replace("Recovery objectives: RTO ", "RTO is ")
        assert _norm(text) in transcript, row                    # the speaker's own words, nothing added


def test_the_card_renders_with_a_source_column(tripwise_card):
    _, rendered, _ = tripwise_card
    table = rendered["blocks"][0]
    assert table["type"] == "DecisionTable"
    assert table["columns"] == ["Situation", "What to do", "Source"]


def test_the_card_comes_first_and_its_sources_are_numbered_first(tripwise_card):
    # The page a new owner opens during an incident; it sat on page 9 to 11.
    _, rendered, order = tripwise_card
    assert order[0] == "quick_reference"
    assert order.count("quick_reference") == 1
    numbers = [n for refs in rendered["blocks"][0].get("sources") or [] for n in refs]
    assert numbers and min(numbers) == 1                     # source numbers follow the printed order


def test_lead_with_quick_reference_moves_only_the_card():
    from knowledge.knowledge_builder import lead_with_quick_reference
    sections = [{"section_id": s} for s in ("system_overview", "unmapped_findings", "kt_coverage", "quick_reference")]
    lead_with_quick_reference(sections)
    assert [s["section_id"] for s in sections] == ["quick_reference", "system_overview", "unmapped_findings", "kt_coverage"]
    without = [{"section_id": "system_overview"}, {"section_id": "kt_coverage"}]
    lead_with_quick_reference(without)
    assert [s["section_id"] for s in without] == ["system_overview", "kt_coverage"]
    lead_with_quick_reference(None)                          # no rendered document: nothing to do


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


def test_a_long_row_keeps_whole_sentences_and_points_to_its_section():
    # A recorded KT's rollback row quoted 120 words of speech on the card.
    ramble = ("We have dev staging and prod and you can see the status of them being synced or out of sync here and "
              "if something is wrong rolling back is done from the same screen and then we decide what to do next "
              "with the team")
    rows = _card(_section("deployment_and_rollback", "Deployment & Rollback",
                          [f"To roll back, redeploy the previous release from ArgoCD. {ramble}."]))
    row = next(r for r in rows if r["Situation"].startswith("A release misbehaves"))
    assert row["What to do"] == "To roll back, redeploy the previous release from ArgoCD. (More in its section.)"
    assert row["Source"] == "Deployment & Rollback"


def test_a_long_safety_row_is_never_shortened():
    # A rule's second sentence can carry its exception or a second window.
    freeze = ("There is a change freeze during the last week of every quarter while finance closes the books, "
              "and nothing goes to production then. We also avoid deploying on Fridays after two in the afternoon "
              "because nobody from the platform team is around over the weekend to watch the release.")
    rows = _card(_section("known_bad_days", "Operational Calendar", [freeze]))
    row = next(r for r in rows if r["Situation"] == "Do not deploy")
    assert len(row["What to do"].split()) > 45
    assert row["What to do"] == freeze


def test_the_step_said_after_the_checks_is_kept_on_the_card():
    # A user's KT (25 Oct): "Check connection-pool usage and application errors
    # before restarting services." The card said no fix was stated.
    from knowledge.knowledge_builder import _known_failures

    said = ["A common issue occurs when database connections are exhausted during examination periods.",
            "Check connection-pool usage and application errors before restarting services."]
    section = {"coverage_content": said, "_structured": {"failures": [
        {"symptom": "Database connections exhausted", "first_checks": "Connection-pool usage; application errors"},
        {"symptom": "Disk full", "first_checks": "Disk usage"}]}}
    rows = dict(_known_failures(section))
    assert rows["Database connections exhausted"] == (
        "First check: Connection-pool usage; application errors, before restarting services. "
        "No other fix was stated in the KT; ask the outgoing owner.")
    assert rows["Disk full"] == "First check: Disk usage. No fix was stated in the KT; ask the outgoing owner."


def test_a_failure_said_to_occur_when_is_named_on_the_card():
    # A user's KT (25 Oct), without an LLM: "A common issue occurs when ..." was
    # not read as naming a failure, and its check was carded as "Where to look first".
    from knowledge.knowledge_builder import _known_failures

    said = ["A common issue occurs when database connections are exhausted during examination periods.",
            "Check connection-pool usage and application errors before restarting services."]
    assert _known_failures({"coverage_content": said}) == [
        ("Database connections are exhausted during examination periods",
         "Check connection-pool usage and application errors before restarting services.")]
