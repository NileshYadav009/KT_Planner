"""P0-11: questions are not facts, and "we don't have that" is a gap, not
coverage. Before: questions became table values ("Staging -> Is there a
staging environment?"), answers lost their topic, and "there is no DR plan"
counted as disaster-recovery coverage."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dialogue import is_gap_statement, is_question, merge_question_answers


@pytest.mark.parametrize("text,gap", [
    ("Do we have a disaster recovery plan? No, honestly there is no DR plan and nobody has tested restores.", True),
    ("Is there a staging environment? Not really, we test directly in production behind a feature flag.", True),
    ("What is the RTO? We never agreed one with the business.", True),
    ("Who do I escalate to? I am not sure anymore, the old manager left.", True),
    ("Is the database backed up? I think so, but I have never checked.", True),
    ("There's no runbook for this.", True),
    # Operational rules and facts that merely contain negative words.
    ("Never delete items from the bookings table by hand.", False),
    ("We never deploy on Fridays.", False),
    ("There is no downtime during deploys because of blue-green.", False),
    ("Still open: the cost review for BigQuery storage has no owner yet.", False),
    ("Is the database backed up? Yes, nightly snapshots to S3.", False),
    ("If you are not sure about a change, involve the platform owner.", False),
])
def test_stated_gaps_are_told_apart_from_rules_and_facts(text, gap):
    assert is_gap_statement(text) is gap


def test_questions_are_merged_with_their_answers():
    texts = ["Is there a staging environment?", "Not really, we test in production.",
             "Who owns billing?", "What about alerts?", "Datadog pages us."]
    merged = merge_question_answers(texts, lambda t: t, lambda a, b: f"{a} {b}")
    assert merged == ["Is there a staging environment? Not really, we test in production.",
                      "Who owns billing?",                     # followed by another question: unanswered
                      "What about alerts? Datadog pages us."]
    assert is_question("Who owns billing?")


DIALOGUE = """Thanks for taking over Pinecart, the checkout service on Google Kubernetes Engine.
Do we have a disaster recovery plan? No, honestly there is no DR plan and nobody has tested restores.
Is there a staging environment? Not really, we test directly in production behind a feature flag.
What is the RTO? We never agreed one with the business.
Who do I escalate to? I am not sure anymore, the old manager left.
Is the database backed up? I think so, but I have never checked.
"""


@pytest.fixture(scope="module")
def dialogue_result():
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
        yield pipeline.run_kt_pipeline("dialogue-0001", clean_transcript(DIALOGUE))
    finally:
        pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE = saved


def _table_cells(result):
    for sec in result["knowledge_object"]["rendered_sections"]:
        for block in sec["blocks"]:
            if block["type"] in ("DecisionTable", "OwnershipTable", "TechnologyGrid") and sec["section_id"] != "kt_coverage":
                for row in block.get("rows") or []:
                    yield sec["section_id"], " | ".join(str(v) for v in row.values())


def test_no_question_or_stated_gap_appears_as_a_table_value(dialogue_result):
    bad = [(sid, cells) for sid, cells in _table_cells(dialogue_result) if "?" in cells or "not sure" in cells.lower()]
    assert bad == []


def test_negated_topics_are_reported_as_stated_gaps(dialogue_result):
    rows = {r["Domain"]: r for sec in dialogue_result["knowledge_object"]["sections"] if sec["id"] == "kt_coverage"
            for r in sec["_coverage_rows"]}
    dr = rows["Disaster Recovery"]
    assert dr["Coverage"] == "Missing" and "stated as not in place" in dr["Assessment"]
    assert rows["Environments"]["Coverage"] == "Missing"


def test_answers_keep_their_topic_and_the_intro_is_the_overview(dialogue_result):
    cov = dialogue_result["coverage"]
    texts = lambda sid: " ".join(s["text"] for s in cov[sid]["sentences"])
    assert "We never agreed one with the business" in texts("disaster_recovery")
    assert "Thanks for taking over Pinecart" in texts("system_overview")


def test_quick_reference_does_not_quote_a_gap_as_guidance(dialogue_result):
    rendered = json.dumps([s for s in dialogue_result["knowledge_object"]["rendered_sections"]
                           if s["section_id"] == "quick_reference"])
    assert "not sure anymore" not in rendered
