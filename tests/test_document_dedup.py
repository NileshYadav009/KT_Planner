"""P1-10: one home per fact. Before, Tribal Knowledge copied every Danger
Zones line, Architecture Details repeated System Overview word for word,
and a transcript that said something three times printed it three times."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from knowledge.document_dedup import WHERE_COLUMN, dedupe_rendered_sections, excerpt


def _narrative(*paragraphs, title="Notes"):
    return {"type": "NarrativeBlock", "title": title, "paragraphs": list(paragraphs)}


def _section(sid, title, *blocks):
    return {"section_id": sid, "section_title": title, "blocks": list(blocks)}


def _paragraphs(section):
    return [p for b in section["blocks"] if b["type"] == "NarrativeBlock" for p in b["paragraphs"]]


def test_a_later_copy_is_removed_and_the_first_kept():
    overview = _section("system_overview", "System Overview",
                        _narrative("Bookings are written to DynamoDB and payments go through Stripe. TripWise serves Europe."))
    arch = _section("architecture_reference", "Architecture Reference",
                    _narrative("Bookings are written to DynamoDB and payments go through Stripe. "
                               "The API runs on Amazon ECS with Fargate.", title="Architecture Details"))
    stats = dedupe_rendered_sections([overview, arch])
    assert _paragraphs(overview) == ["Bookings are written to DynamoDB and payments go through Stripe. TripWise serves Europe."]
    assert _paragraphs(arch) == ["The API runs on Amazon ECS with Fargate."]
    assert stats["removed"] == 1


def test_a_sentence_said_three_times_is_printed_once():
    said = "One recurring problem is DynamoDB throttling during flash sales."
    failures = _section("common_failures", "Common Failures", _narrative(said, said, said + " The fix is on-demand capacity."))
    dedupe_rendered_sections([failures])
    assert _paragraphs(failures) == [said, "The fix is on-demand capacity."]


def test_tables_and_lists_are_never_cut_but_count_as_shown():
    owners = _section("ownership_escalation", "Ownership", {
        "type": "OwnershipTable", "title": "Owners",
        "rows": [{"role": "Backend", "team": "Our squad owns the backend and the release pipeline."}]})
    later = _section("open_responsibilities", "Open Responsibilities",
                     _narrative("Our squad owns the backend and the release pipeline. Week three: own releases."))
    dedupe_rendered_sections([owners, later])
    assert owners["blocks"][0]["rows"][0]["team"] == "Our squad owns the backend and the release pipeline."
    assert _paragraphs(later) == ["Week three: own releases."]


@pytest.mark.parametrize("first,second", [
    ("RTO is four hours for the primary booking database.", "RTO is two hours for the primary booking database."),
    ("We deploy to production on Fridays after the change review.",
     "We never deploy to production on Fridays after the change review."),
    ("The retry job runs every 15 minutes during the night.", "The retry job runs every 30 minutes during the night."),
])
def test_different_numbers_or_a_negation_are_different_facts(first, second):
    a = _section("disaster_recovery", "DR", _narrative(first))
    b = _section("known_bad_days", "Calendar", _narrative(second))
    dedupe_rendered_sections([a, b])
    assert _paragraphs(b) == [second]


def test_the_same_sentence_reworded_by_punctuation_or_case_is_one_fact():
    a = _section("architecture_reference", "Architecture", _narrative("Redis is used for caching and short-lived session data."))
    b = _section("environments", "Environments", _narrative("redis is used for caching, and short lived session data"))
    dedupe_rendered_sections([a, b])
    assert b["blocks"] == [] or _paragraphs(b) == ["Everything said about this was already shown under Architecture."]


def test_text_that_quotes_on_purpose_is_left_alone():
    rule = "Never delete items from the bookings table by hand, since refunds depend on that history."
    danger = _section("danger_zones", "Danger Zones", _narrative(rule))
    calendar = _section("known_bad_days", "Calendar", {"type": "WarningBlock",
                                                         "title": "Possible conflict — confirm with the outgoing owner",
                                                         "warnings": [rule]}, _narrative("Avoid the July sale for deploys."))
    arch = _section("architecture_reference", "Architecture", {
        "type": "DecisionTable", "title": "Connections stated in the KT", "columns": ["Said in the KT"],
        "rows": [{"Said in the KT": rule}]})
    card = _section("quick_reference", "Quick Reference", {
        "type": "DecisionTable", "title": "Card", "columns": ["What to do"], "rows": [{"What to do": rule}]})
    dedupe_rendered_sections([danger, calendar, arch, card])
    assert calendar["blocks"][0]["warnings"] == [rule]
    assert arch["blocks"][0]["rows"][0]["Said in the KT"] == rule
    assert card["blocks"][0]["rows"][0]["What to do"] == rule


def test_tribal_knowledge_points_to_the_full_statement_instead_of_copying_it():
    rule = "Never delete items from the bookings table by hand, since refunds depend on that history."
    danger = _section("danger_zones", "Danger Zones", _narrative(rule))
    tribal = _section("tribal_knowledge", "Tribal Knowledge", {
        "type": "DecisionTable", "title": "Non-obvious operational knowledge", "columns": ["Knowledge", "Classification"],
        "rows": [{"Knowledge": rule, "Classification": "Safety-critical"},
                 {"Knowledge": "CloudFront invalidation can take longer than expected during large releases.",
                  "Classification": "Operational"}]})
    stats = dedupe_rendered_sections([danger, tribal])
    rows = tribal["blocks"][0]["rows"]
    assert rows[0]["Knowledge"] == "Never delete items from the bookings table by…"
    assert rows[0][WHERE_COLUMN] == "Danger Zones"
    assert rows[1]["Knowledge"].startswith("CloudFront invalidation") and rows[1][WHERE_COLUMN] == "Only here"
    assert tribal["blocks"][0]["columns"] == ["Knowledge", "Classification", WHERE_COLUMN]
    assert stats["referenced"] == 1


def test_danger_zones_own_warning_block_is_a_home_for_the_digest():
    rule = "Never purge the dead letter queue, because those are undelivered critical results."
    danger = _section("danger_zones", "Danger Zones", {"type": "WarningBlock", "title": "Danger Zones", "warnings": [rule]})
    tribal = _section("tribal_knowledge", "Tribal Knowledge", {
        "type": "DecisionTable", "title": "Digest", "columns": ["Knowledge", "Classification"],
        "rows": [{"Knowledge": rule, "Classification": "Safety-critical"}]})
    dedupe_rendered_sections([danger, tribal])
    assert danger["blocks"][0]["warnings"] == [rule]
    assert tribal["blocks"][0]["rows"][0][WHERE_COLUMN] == "Danger Zones"


def test_placeholder_text_is_not_a_fact():
    note = "This section was not covered in the KT session. Flag it for follow-up with the outgoing owner."
    a = _section("security_controls", "Security", _narrative(note))
    b = _section("signoff", "Sign-off", _narrative(note))
    dedupe_rendered_sections([a, b])
    assert _paragraphs(b) == [note]


def test_excerpt():
    assert excerpt("Never delete items from the bookings table by hand, since refunds depend on that history.") \
        == "Never delete items from the bookings table by…"
    assert excerpt("Datadog is our monitoring tool.") == "Datadog is our…"
    # Never a whole sentence, even when the first sentence is short.
    first_sentence_is_short = "Another common issue is missing Kubernetes secrets. If pods fail, check secret references."
    assert excerpt(first_sentence_is_short) == "Another common issue is missing…"


# --------------------------------------------------------------------------
# Golden: real KTs end to end
# --------------------------------------------------------------------------

_EXEMPT_BLOCKS = ("WarningBlock", "ImageBlock", "DiagramBlock")
_PLACEHOLDER = re.compile(r"not covered (?:in|during) the kt|flag it for follow", re.IGNORECASE)


def _repeated_sentences(result):
    seen, repeats = {}, []
    for sec in result["knowledge_object"]["rendered_sections"]:
        if sec["section_id"] in ("quick_reference", "kt_coverage"):
            continue
        for block in sec.get("blocks", []):
            if block.get("type") in _EXEMPT_BLOCKS or block.get("title") == "Connections stated in the KT":
                continue
            # What a reader sees: a table's displayed columns only, less the
            # digest's "Where it is" pointer (a section name, not a fact).
            columns = [c for c in block.get("columns") or [] if c != WHERE_COLUMN]
            texts = block.get("paragraphs") or block.get("warnings") \
                or [str(r.get(c, "")) for r in block.get("rows") or [] for c in (columns or list(r))] \
                or [str(i) for i in block.get("items") or []]
            for text in texts:
                for sentence in re.split(r"(?<=[.!?])\s+", str(text)):
                    key = re.sub(r"[^a-z0-9 ]+", "", sentence.lower()).strip()
                    if len(key.split()) < 5 or _PLACEHOLDER.search(sentence):
                        continue
                    if key in seen:
                        repeats.append((sentence, seen[key], sec["section_id"]))
                    seen.setdefault(key, sec["section_id"])
    return repeats


@pytest.fixture(scope="module")
def golden_results():
    import ai
    import pipeline
    from context_mapper import ContextMappingPipeline
    from devops_transcription import clean_transcript
    from kt_schema_loader import SCHEMA
    from test_ecommerce_kt import TRANSCRIPT as ECOMMERCE
    from test_incident_card import TRIPWISE

    saved = (pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE)
    pipeline.get_llm_provider = ai.get_llm_provider = lambda: None
    pipeline.record_candidates = lambda *a, **k: None
    pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
    results = {}
    try:
        for name, text in (("tripwise_x3", "\n\n".join([TRIPWISE] * 3)), ("ecommerce", ECOMMERCE)):
            results[name] = pipeline.run_kt_pipeline(f"dedup-golden-{name}", clean_transcript(text))
            pipeline.JOB_STORE.delete(f"dedup-golden-{name}")
    finally:
        pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE = saved
    return results


@pytest.mark.parametrize("name", ["tripwise_x3", "ecommerce"])
def test_no_fact_sentence_renders_more_than_once(golden_results, name):
    assert _repeated_sentences(golden_results[name]) == []


@pytest.mark.parametrize("name", ["tripwise_x3", "ecommerce"])
def test_nothing_said_is_lost_from_the_document(golden_results, name):
    ko = golden_results[name]["knowledge_object"]
    assert ko["_knowledge_coverage_summary"]["recovered"] == 0      # no sentence had to be put back
    assert ko["_dedup"]["removed"] > 0
