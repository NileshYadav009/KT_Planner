"""P1-2: every fact in the document points to the transcript sentence(s) it
came from and when it was said; inferred values are never presented as
stated."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from knowledge.evidence import attach_evidence, evidence_pairs, format_time, transcript_sentences
from knowledge.knowledge_builder import INFERRED_MARKER
from renderers.blocks.common import MENTIONED_ELSEWHERE_MESSAGE, NOT_COVERED_MESSAGE

SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "Welcome to the KT for the payments platform."},
    {"start": 4.0, "end": 9.5, "text": "The payments API runs on EKS in us-east-1 and stores ledgers in Aurora"},
    {"start": 9.5, "end": 12.0, "text": "PostgreSQL. Alerts go to PagerDuty, and the on-call rotation is weekly."},
    {"start": 12.0, "end": 18.0, "text": "Never restart the ledger writer during the 2 AM batch window."},
]


def _ko(*blocks, section_id="architecture_reference"):
    return {"rendered_sections": [{"section_id": section_id, "section_title": "S", "blocks": list(blocks)}]}


def test_sentences_keep_the_time_they_were_said_in_the_original_recording():
    sentences = transcript_sentences("", SEGMENTS, offset=30.0)
    quotes = [s["quote"] for s in sentences]
    assert "The payments API runs on EKS in us-east-1 and stores ledgers in Aurora PostgreSQL." in quotes
    spanning = next(s for s in sentences if s["quote"].startswith("The payments API"))
    assert (spanning["start"], spanning["end"]) == (34.0, 42.0)          # starts in one segment, ends in the next
    assert format_time(spanning["start"]) == "00:34" and format_time(3725) == "1:02:05"


def test_a_pasted_transcript_has_sources_without_times():
    sentences = transcript_sentences("Alerts go to PagerDuty. The rotation is weekly.")
    assert [s["start"] for s in sentences] == [None, None]
    assert format_time(None) == ""


def test_each_fact_points_to_the_sentence_that_states_it():
    ko = _ko(
        {"type": "NarrativeBlock", "title": "S", "paragraphs": [
            "The payments API runs on EKS in us-east-1.",                       # part of a longer sentence
            "Alerts are routed to PagerDuty with a weekly on-call rotation.",    # a paraphrase
        ]},
        {"type": "WarningBlock", "title": "Danger", "warnings": ["Never restart the ledger writer in the 2 AM batch window."]},
        {"type": "TechnologyGrid", "title": "Stack", "rows": [{"label": "Database", "value": "Aurora PostgreSQL"}]},
    )
    stats = attach_evidence(ko, transcript_sentences("", SEGMENTS))
    assert stats == {"units": 4, "stated": 4, "inferred": 0, "unsourced": 0}
    pairs = dict((text, quotes) for text, quotes in evidence_pairs(ko))
    assert pairs["Alerts are routed to PagerDuty with a weekly on-call rotation."] == [
        "Alerts go to PagerDuty, and the on-call rotation is weekly."]
    assert pairs["Never restart the ledger writer in the 2 AM batch window."] == [
        "Never restart the ledger writer during the 2 AM batch window."]
    # Sources are numbered in document order, each sentence once.
    assert [s["id"] for s in ko["sources"]] == list(range(1, len(ko["sources"]) + 1))
    assert ko["rendered_sections"][0]["blocks"][0]["sources"][0] == [1]


def test_a_number_the_session_never_said_is_not_sourced():
    ko = _ko({"type": "NarrativeBlock", "title": "S", "paragraphs": [
        "Never restart the ledger writer during the 4 AM batch window."]})
    assert attach_evidence(ko, transcript_sentences("", SEGMENTS))["unsourced"] == 1
    assert ko["rendered_sections"][0]["blocks"][0]["sources"] == [[]]


def test_inferred_values_and_status_lines_are_never_linked_as_stated():
    ko = _ko({"type": "NarrativeBlock", "title": "S", "paragraphs": [
        "The payments API runs on EKS" + INFERRED_MARKER, NOT_COVERED_MESSAGE, MENTIONED_ELSEWHERE_MESSAGE]})
    stats = attach_evidence(ko, transcript_sentences("", SEGMENTS))
    assert stats == {"units": 1, "stated": 0, "inferred": 1, "unsourced": 0}
    assert ko["rendered_sections"][0]["blocks"][0]["sources"] == [[], [], []]


def test_blocks_carry_only_ids_never_transcript_text():
    ko = _ko({"type": "ChecklistBlock", "title": "S", "items": ["Alerts go to PagerDuty."]})
    attach_evidence(ko, transcript_sentences("", SEGMENTS))
    block = ko["rendered_sections"][0]["blocks"][0]
    assert all(isinstance(i, int) for refs in block["sources"] for i in refs)


def test_the_pdf_shows_a_reference_after_each_fact_and_lists_the_sources():
    from pdf_rendering import render_pdf_html

    ko = _ko({"type": "ChecklistBlock", "title": "Alerts", "items": ["Alerts go to PagerDuty."]})
    attach_evidence(ko, transcript_sentences("", SEGMENTS, offset=60.0))
    html = render_pdf_html("Payments", "job", ko["rendered_sections"], {}, "6 October 2026", sources=ko["sources"])
    assert '<sup class="src-ref"><a href="#src-1">1</a></sup>' in html
    assert '<li id="src-1"><span class="src-n">1</span><span class="src-t">01:09</span>' in html
    assert "Alerts go to PagerDuty, and the on-call rotation is weekly." in html
    assert 'href="#sources"' in html                                       # in the table of contents


GCP = [{"start": float(i * 5), "end": float(i * 5 + 5), "text": t} for i, t in enumerate([
    "Hi everyone, today I will be handing over the GCP data platform.",
    "GKE is used for Kubernetes workloads.",
    "Grafana provides dashboards, and PagerDuty is used for alerting.",
    "One common production problem is Pub/Sub backlog.",
    "The rto is four hours, and the rpo is one hour.",
    "That concludes the GCP data platform.",
])]


def test_the_documents_wording_of_a_fact_still_finds_what_was_said():
    ko = {"system_name": "GCP Data Platform", "rendered_sections": [
        {"section_id": "system_overview", "section_title": "System Overview", "blocks": [
            {"type": "TechnologyGrid", "title": "Stack", "rows": [
                {"label": "Compute", "value": "Google Kubernetes Engine"},            # the speaker said "GKE"
                {"label": "Region", "value": "Not discussed"}]}]},                     # a placeholder, not a fact
        {"section_id": "disaster_recovery", "section_title": "Disaster Recovery", "blocks": [
            {"type": "NarrativeBlock", "title": "DR", "paragraphs": ["RTO (Recovery Time Objective): four hours"]}]},
        {"section_id": "ownership_escalation", "section_title": "Ownership", "blocks": [
            {"type": "OwnershipTable", "title": "Owners", "rows": [{"role": "On-call tool", "team": "PagerDuty"}]}]},
        {"section_id": "quick_reference", "section_title": "Quick Reference", "blocks": [
            {"type": "DecisionTable", "title": "Card", "columns": ["Situation", "What to do", "Where it is"], "rows": [
                {"Situation": "Pub/Sub backlog", "What to do": "No fix was stated in the KT. Ask the outgoing owner.",
                 "Where it is": "Disaster Recovery"}]}]},
        {"section_id": "architecture_reference", "section_title": "Architecture", "blocks": [
            {"type": "DecisionTable", "title": "Connections", "columns": ["Connection", "Said"], "rows": [
                {"Connection": "GCP Data Platform runs on Google Kubernetes Engine",
                 "Said": "GKE is used for Kubernetes workloads."}]}]},
    ]}
    stats = attach_evidence(ko, transcript_sentences("", GCP))
    assert stats == {"units": 5, "stated": 5, "inferred": 0, "unsourced": 0}
    pairs = dict(evidence_pairs(ko))
    assert pairs["Google Kubernetes Engine"] == ["GKE is used for Kubernetes workloads."]
    assert pairs["RTO (Recovery Time Objective): four hours"] == ["The rto is four hours, and the rpo is one hour."]
    assert pairs["On-call tool PagerDuty"] == ["Grafana provides dashboards, and PagerDuty is used for alerting."]
    # The section name in the row is a pointer; only the failure is a statement.
    assert pairs[next(k for k in pairs if k.startswith("Pub/Sub backlog"))] == ["One common production problem is Pub/Sub backlog."]
    # The system's own name in a row does not pull in the sentences that name the platform.
    assert pairs[next(k for k in pairs if k.startswith("GCP Data Platform runs"))] == ["GKE is used for Kubernetes workloads."]
    assert not any(q.startswith("That concludes") for qs in pairs.values() for q in qs)
