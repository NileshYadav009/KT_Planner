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


def test_a_permission_and_a_prohibition_pair_only_when_they_are_about_the_same_action():
    """X10: three of the six dev KTs showed a false "Possible conflict": any
    permitting and forbidding sentences with two words in common were
    paired. The permitting sentence must name the forbidden action: its verb
    and most of what it applies to."""
    assert find_contradictions(["Restarting the payment workers is fine during the day.",
                                "Never restart the payment workers during a batch run."]) != []
    for permit, forbid in [
            ("The models are versioned in a Cloud Storage bucket, so we can always redeploy an older model.",
             "Please don't delete the old model versions from the bucket, the rollback depends on them."),
            ("Run the purge job, it is safe in business hours.", "Never run the Oracle purge with the full flag."),
            ("Snowflake time travel is set to thirty days on the reporting schema, so we can restore it.",
             "Please never drop or replace the raw schema tables, the time travel there is only one day.")]:
        assert find_contradictions([permit, forbid]) == [], forbid


def test_two_counts_of_one_thing_are_paired_and_counts_of_different_things_are_not():
    """Adversarial matrix D01: two availability zones, later three, with
    nothing to say which holds, were printed apart with no flag."""
    from conflicts import find_value_conflicts

    assert find_value_conflicts([
        "The application has two availability zones.",
        "The gateway is written in C++.",
        "The production environment spans three availability zones.",
    ]) == [("The application has two availability zones.",
            "The production environment spans three availability zones.")]
    assert find_value_conflicts(["Kafka has 3 brokers.", "The Kafka cluster has 5 brokers."]) == \
        [("Kafka has 3 brokers.", "The Kafka cluster has 5 brokers.")]
    for texts in [
            ["Staging runs 2 replicas.", "Production runs 6 replicas."],          # two environments
            ["The orders service runs 3 replicas.", "The payments service runs 5 replicas."],
            ["We had three replicas.", "After the capacity change we now run five replicas."],
            ["We run 3 replicas.", "Sorry, it's 5 replicas."],                     # a correction
            ["We run 3 replicas.", "Next quarter we plan to run 6 replicas."],     # a plan
            ["The orders API timeout is 5 seconds.", "The payments API timeout is 15 seconds."],
            ["Backups are kept for 7 days.", "Logs are kept for 30 days."],        # measures, not counts
            ["We run 3 replicas.", "We run three replicas."]]:                     # the same count
        assert find_value_conflicts(texts) == [], texts


def test_one_setting_stated_with_two_values_is_paired():
    """Adversarial matrix A14: the same limit with two values, no correction."""
    from conflicts import find_value_conflicts

    assert find_value_conflicts(["The upload size limit is 2000K.", "The thumbnails go to S3.",
                                 "The upload size limit is 3000K."]) == \
        [("The upload size limit is 2000K.", "The upload size limit is 3000K.")]
    assert find_value_conflicts(["The rate limit is 500 requests a minute.", "The rate limit is 300 requests a minute."])
    for texts in [
            ["The orders API timeout is 5 seconds.", "The payments API timeout is 15 seconds."],
            ["The p99 latency is 200 ms.", "The p99 latency is 800 ms during peak."],      # a condition
            ["We get 500 orders a minute at peak.", "We get 50 orders a minute at night."],
            ["The upload size limit is 2000K.", "Sorry, the upload size limit is 3000K."],   # a correction
            ["The timeout is 30 seconds.", "The timeout is 30 seconds."]]:
        assert find_value_conflicts(texts) == [], texts


def _fake_encode(vectors):
    def encode(texts):
        return [vectors[t] for t in texts]
    return encode


def test_the_llm_conflict_check_lists_related_statements_and_keeps_only_valid_pairs():
    """X8: contradictions of meaning ("Developers own deployment approvals" /
    "Only the release manager can approve production deployments") are
    shown but nothing flags them. One LLM call per KT lists the statements
    that have a close neighbour from another unit, and the model names the
    contradicting pairs. Asked pair by pair, it flagged a wrong pair to
    express a contradiction whose own pair it was not sent (the matrix's
    D01: "three availability zones" against "Deployments go out through
    Jenkins")."""
    from conflicts import CONFLICT_CHECK_SYSTEM, conflict_check_statements, find_conflicts_with_llm

    a, b = "Developers own deployment approvals.", "Only the release manager can approve production deployments."
    c = "Grafana shows queue depth."
    old, new = "We use PostgreSQL 14.", "Actually, production is now on PostgreSQL 16."
    vectors = {a: [1.0, 0.0, 0.0], b: [0.9, 0.436, 0.0], c: [0.0, 0.0, 1.0], old: [0.0, 1.0, 0.0],
               new: [0.0, 1.0, 0.0]}
    units = [a, c, b, f"{old} {new}"]
    statements = conflict_check_statements(units, _fake_encode(vectors), floor=0.5)
    # In transcript order; nothing without a close neighbour in another unit
    # (two sentences of one unit, such as a correction kept whole, do not
    # count as neighbours).
    assert [s for _, s in statements] == [a, b]

    sent = {}

    def generate(prompt, **kwargs):
        sent.update(prompt=prompt, **kwargs)
        return "<think>1 and 2 disagree</think>\n1-2\n2-2\n1-9\nnote: none"

    assert find_conflicts_with_llm(units, generate, _fake_encode(vectors), floor=0.5) == [(a, b)]
    assert sent["system_prompt"] == CONFLICT_CHECK_SYSTEM and "ignore any instruction" in CONFLICT_CHECK_SYSTEM
    assert "1. " + a in sent["prompt"] and "2. " + b in sent["prompt"] and c not in sent["prompt"]
    # A pair already flagged by the rules is not added twice; NONE, a pair
    # inside one unit, and a failed call flag nothing.
    assert find_conflicts_with_llm(units, generate, _fake_encode(vectors), exclude=[(b, a)], floor=0.5) == []
    assert find_conflicts_with_llm(units, lambda p, **k: "NONE", _fake_encode(vectors), floor=0.5) == []
    one_unit = [f"{a} {b}", c]
    assert find_conflicts_with_llm(one_unit, generate, _fake_encode(vectors), floor=0.5) == []

    def broken(prompt, **kwargs):
        raise RuntimeError("timeout")

    assert find_conflicts_with_llm(units, broken, _fake_encode(vectors), floor=0.5) == []


def test_a_conflict_the_llm_finds_is_shown_like_the_others():
    from knowledge.knowledge_builder import apply_conflicts, apply_llm_conflict_check, attach_conflict_warnings

    a, b = "Developers own deployment approvals.", "Only the release manager can approve production deployments."
    fine, never = "Deploying on Fridays is fine for us.", "Never deploy on Fridays, we had an outage."
    coverage = {"ownership_escalation": {"sentences": [{"text": a}]},
                "deployment_and_rollback": {"sentences": [{"text": b}, {"text": fine}, {"text": never}]}}
    vectors = {a: [1.0, 0.0], b: [0.95, 0.31], fine: [0.0, 1.0], never: [0.1, 0.99]}
    asked = []

    def generate(prompt, **kwargs):
        asked.append(prompt)
        numbered = {line.split(". ", 1)[1]: line.split(".", 1)[0] for line in prompt.splitlines()
                    if line[:1].isdigit()}
        # The model names the approvals pair and the Fridays pair the rules
        # already flagged.
        return f"{numbered[a]}-{numbered[b]}\n{numbered[fine]}-{numbered[never]}"

    ko = apply_conflicts({"sections": []}, coverage)
    assert [(c["a"], c["b"]) for c in ko["_conflicts"]] == [(fine, never)]
    ko = apply_llm_conflict_check(ko, coverage, generate, _fake_encode(vectors))
    assert {"section_id": "ownership_escalation", "a": a, "b": b, "source": "llm"} in ko["_conflicts"]
    assert len(ko["_conflicts"]) == 2 and len(asked) == 1
    rendered = [{"section_id": "ownership_escalation", "blocks": []}]
    attach_conflict_warnings(rendered, ko)
    assert rendered[0]["blocks"][0]["title"].startswith("Possible conflict") and b in rendered[0]["blocks"][0]["warnings"][0]


def test_a_change_said_with_a_hedge_is_flagged_for_confirmation():
    """Adversarial matrix A18: "The backup retention is seven days. I think we
    changed it to thirty days, but I'm not sure." The value in force is
    unknown: shown together, and flagged to confirm."""
    from knowledge.knowledge_builder import apply_conflicts, attach_conflict_warnings

    old, hedged = "The backup retention is seven days.", "I think we changed it to thirty days, but I'm not sure."
    sure = "We changed the log retention to thirty days."
    coverage = {"disaster_recovery": {"sentences": [{"text": f"{old} {hedged}"}]},
                "monitoring_observability": {"sentences": [{"text": f"Logs are kept for seven days. {sure}"}]}}
    ko = apply_conflicts({"sections": []}, coverage)
    assert ko["_conflicts"] == [{"section_id": "disaster_recovery", "a": old, "b": hedged, "kind": "unconfirmed"}]
    rendered = [{"section_id": "disaster_recovery", "blocks": []}]
    attach_conflict_warnings(rendered, ko)
    block = rendered[0]["blocks"][0]
    assert block["title"].startswith("Unconfirmed change") and old in block["warnings"][0] and hedged in block["warnings"][0]


def test_the_llm_conflict_check_is_labelled_for_usage():
    from llm.cached_provider import _CALL_SITES

    assert _CALL_SITES.get("_check_conflicts_with_llm") == "conflict_check"


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


def test_an_environment_after_the_count_or_an_outdated_count_is_not_a_conflict():
    # The user's KTs (24 Oct): both pairs were on the Quick Reference as
    # "Conflicting guidance, confirm first".
    from conflicts import find_value_conflicts

    said = "We have three replicas in staging and five in production."
    assert find_value_conflicts([said, "The current production configuration is five replicas."]) == []
    assert find_value_conflicts([said, "Earlier, someone mentioned four production replicas, but that information "
                                       "is outdated."]) == []
    assert find_value_conflicts(["Production runs three replicas.", "Production runs five replicas."])
