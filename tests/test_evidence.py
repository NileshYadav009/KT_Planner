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


def test_a_reworded_paragraph_is_sourced_sentence_by_sentence():
    segments = [{"start": float(i), "end": float(i + 1), "text": t} for i, t in enumerate([
        "so today we ran 64 pipelines across 7 repositories right.",
        "The blocked by policy pipelines are 2 this week.",
        "The median duration to run a pipeline is 9 minutes.",
        "Rollbacks go through the release manager on call.",
        "Canary deploys cover 10 percent of the traffic first."])]
    edited = ("Today we ran 64 pipelines across 7 repositories. Pipelines blocked by policy are 2 this week. "
              "A pipeline's median duration is 9 minutes. Rollbacks go through the on-call release manager. "
              "Canary deploys cover 10 percent of traffic first.")
    ko = _ko({"type": "NarrativeBlock", "title": "S", "paragraphs": [edited]})
    assert attach_evidence(ko, transcript_sentences("", segments))["unsourced"] == 0
    assert ko["rendered_sections"][0]["blocks"][0]["sources"] == [[1, 2, 3, 4, 5]]
    # One sentence nobody said leaves the whole paragraph unsourced.
    ko = _ko({"type": "NarrativeBlock", "title": "S", "paragraphs": [edited + " Failover takes 4 minutes."]})
    assert attach_evidence(ko, transcript_sentences("", segments))["unsourced"] == 1


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


def test_a_fact_with_no_source_is_marked_in_the_pdf_and_gap_notes_are_not_facts():
    """An LLM restatement no sentence supports ('Receive page from PagerDuty')
    printed exactly like a stated fact; a section's note on what the KT left
    out names topics and states nothing."""
    from pdf_rendering import render_pdf_html

    ko = _ko({"type": "ChecklistBlock", "title": "First response", "items": [
        "Alerts go to PagerDuty.", "Rotate the database password from the admin console.",
        "Not covered in the KT session: Fixes or workarounds (confirm with the outgoing owner)."]})
    stats = attach_evidence(ko, transcript_sentences("", SEGMENTS))
    assert (stats["stated"], stats["unsourced"]) == (1, 1)
    assert ko["rendered_sections"][0]["blocks"][0]["unsourced"] == [1]
    html = render_pdf_html("Payments", "job", ko["rendered_sections"], {}, "6 October 2026", sources=ko["sources"])
    assert html.count('<sup class="src-ref src-none">no source</sup>') == 1


def test_a_numbered_labelled_step_is_linked_to_the_sentence_it_repeats():
    """'1. **Trigger:** <sentence>' had no source: the list's own '1' was
    treated as a number the session must have said."""
    ko = _ko({"type": "ChecklistBlock", "title": "Steps", "items": [
        "1. **Trigger:** Alerts go to PagerDuty, and the on-call rotation is weekly."]})
    stats = attach_evidence(ko, transcript_sentences("", SEGMENTS))
    assert (stats["stated"], stats["unsourced"]) == (1, 0)


def test_an_acronym_is_sourced_by_its_full_name_said_aloud():
    """The cleaner now leaves "recovery time objective" as said (audit 9aak);
    the document's "RTO (Recovery Time Objective): four hours" must still
    point to it."""
    ko = _ko({"type": "NarrativeBlock", "title": "S", "paragraphs": [
        "RTO (Recovery Time Objective): four hours"]}, section_id="disaster_recovery")
    sentences = transcript_sentences("", [{"start": 0.0, "end": 3.0, "text": "The recovery time objective is four hours."}])
    assert attach_evidence(ko, sentences)["unsourced"] == 0
    assert len(ko["rendered_sections"][0]["blocks"][0]["sources"][0]) == 1


def test_a_corrected_number_is_sourced_by_the_sentence_and_its_correction():
    """The document shows "300" (context_mapper.merge_spoken_corrections);
    only the correction says it, and the first sentence says the rest."""
    ko = _ko({"type": "NarrativeBlock", "title": "S", "paragraphs": [
        "The API rate limit is 300 requests a minute per customer."]})
    sentences = transcript_sentences("", [
        {"start": 0.0, "end": 3.0, "text": "The API rate limit is 500 requests a minute per customer."},
        {"start": 3.0, "end": 6.0, "text": "Sorry, not 500, the limit is 300 requests a minute per customer."}])
    assert attach_evidence(ko, sentences)["unsourced"] == 0
    assert ko["rendered_sections"][0]["blocks"][0]["sources"][0] == [1, 2]
    # A number said about something else is still no source: "the burst
    # limit is 300" does not state that the rate limit is 300.
    ko = _ko({"type": "NarrativeBlock", "title": "S", "paragraphs": [
        "The API rate limit is 300 requests a minute per customer."]})
    sentences = transcript_sentences("", [
        {"start": 0.0, "end": 3.0, "text": "The API rate limit is 500 requests a minute per customer."},
        {"start": 3.0, "end": 6.0, "text": "The burst limit is 300 requests a minute."}])
    assert attach_evidence(ko, sentences)["unsourced"] == 1


def test_each_statement_of_a_paragraph_cites_its_own_sentence():
    """A polished paragraph matched as a whole stopped at three sources, or
    at the one sentence that covered most of its words, and a statement in it
    cited nothing (Groq dev run, audit 9aap)."""
    said = ["Overview: Vaultline is the adapter between the mobile banking app and the core banking mainframe.",
            "It translates REST calls into mainframe transactions.",
            "It is used by about one point two million retail customers.",
            "If it is down, customers cannot see balances or make transfers.",
            "MQ channel stops after a mainframe restart.",
            "Fix: restart the channel from the MQ console."]
    sentences = transcript_sentences("", [{"start": float(i), "end": float(i + 1), "text": t} for i, t in enumerate(said)])
    shown = [said[0].replace("Overview: ", "") + " " + " ".join(said[1:4]),
             said[4] + " " + said[5].replace("Fix: restart", "Restart")]
    ko = _ko({"type": "NarrativeBlock", "title": "S", "paragraphs": shown})
    assert attach_evidence(ko, sentences)["unsourced"] == 0
    assert ko["rendered_sections"][0]["blocks"][0]["sources"] == [[1, 2, 3, 4], [5, 6]]


def test_each_statement_cell_of_a_row_cites_its_own_sentence():
    """An issue and its fix in one row cited only the symptom, and the fix
    had no source (Groq dev run, audit 9aap)."""
    said = ["Known problems.", "MQ channel stops after a mainframe restart.",
            "Symptom is every transfer timing out.", "Fix: restart the channel from the MQ console."]
    sentences = transcript_sentences("", [{"start": float(i), "end": float(i + 1), "text": t} for i, t in enumerate(said)])
    ko = _ko({"type": "DecisionTable", "title": "Failures", "columns": ["Issue", "Fix"], "rows": [
        {"Issue": "MQ channel stops after a mainframe restart", "Fix": "Restart the channel from the MQ console"}]},
        section_id="common_failures")
    assert attach_evidence(ko, sentences)["unsourced"] == 0
    pairs = dict(evidence_pairs(ko))
    quotes = pairs[next(iter(pairs))]
    assert "Fix: restart the channel from the MQ console." in quotes and \
        "MQ channel stops after a mainframe restart." in quotes


def test_a_list_of_names_said_in_different_sentences_cites_each():
    # A user's KT (23 Oct): "GitOps / deployment: GitHub Actions; Helm" was
    # printed "no source"; GitHub Actions and Helm were each said in their
    # own sentence.
    said = ["Deployments happen through GitHub Actions.", "The service uses Redis for caching.",
            "If a deployment fails, check the logs first and rollback using Helm."]
    sentences = transcript_sentences("", [{"start": float(i), "end": float(i + 1), "text": t} for i, t in enumerate(said)])
    ko = _ko({"type": "TechnologyGrid", "title": "Technology summary",
              "rows": [{"label": "GitOps / deployment", "value": "GitHub Actions; Helm"}]})
    assert attach_evidence(ko, sentences)["unsourced"] == 0
    quotes = dict(evidence_pairs(ko))
    cited = [q for qs in quotes.values() for q in qs]
    assert said[0] in cited and said[2] in cited and said[1] not in cited


def test_a_quote_or_a_list_of_names_in_a_paragraph_cites_what_was_said():
    # The user's KTs (24 Oct): "Prometheus, Grafana, PagerDuty" and a "Mentioned
    # elsewhere" item quoting a sentence were printed "no source".
    said = ["Prometheus collects application metrics.", "Grafana displays API latency.",
            "PagerDuty alerts the on-call engineer.", "Pipeline failures generate Slack notifications."]
    sentences = transcript_sentences("", [{"start": float(i), "end": float(i + 1), "text": t} for i, t in enumerate(said)])
    names = _ko({"type": "NarrativeBlock", "title": "Architecture Knowledge",
                 "paragraphs": ["Prometheus, Grafana, PagerDuty"]})
    assert attach_evidence(names, sentences)["unsourced"] == 0
    quoted = _ko({"type": "ChecklistBlock", "title": "Mentioned elsewhere in the KT", "items": [
        "Paging and alert routing, under Deployment & Rollback: \u201cPipeline failures generate Slack notifications.\u201d"]},
        section_id="monitoring_observability")
    assert attach_evidence(quoted, sentences)["unsourced"] == 0
    assert [q for qs in dict(evidence_pairs(quoted)).values() for q in qs] == [said[3]]
    # A list with a name nobody said stays unsourced.
    unsaid = _ko({"type": "NarrativeBlock", "title": "Architecture Knowledge", "paragraphs": ["Prometheus, Datadog"]})
    assert attach_evidence(unsaid, sentences)["unsourced"] == 1


def test_a_category_value_is_sourced_by_what_its_label_names():
    # The user's recordings (24 Oct): "Business criticality: High", chosen from
    # "It is business critical because ...", was printed "no source".
    said = ["This platform processes customer orders.",
            "It is business critical because customers cannot place orders when it is down."]
    sentences = transcript_sentences("", [{"start": float(i), "end": float(i + 1), "text": t} for i, t in enumerate(said)])
    ko = _ko({"type": "TechnologyGrid", "title": "Captured knowledge",
              "rows": [{"label": "Business criticality", "value": "High"}]}, section_id="system_overview")
    assert attach_evidence(ko, sentences)["unsourced"] == 0
    assert [q for qs in dict(evidence_pairs(ko)).values() for q in qs] == [said[1]]
    # Said without the label's first word: "mission critical".
    said = ["This is a mission critical banking system.", "Azure SQL is the primary database."]
    sentences = transcript_sentences("", [{"start": float(i), "end": float(i + 1), "text": t} for i, t in enumerate(said)])
    ko = _ko({"type": "TechnologyGrid", "title": "Captured knowledge",
              "rows": [{"label": "Business criticality", "value": "High"}]}, section_id="system_overview")
    assert attach_evidence(ko, sentences)["unsourced"] == 0


def test_a_greeting_is_never_cited_and_table_labels_are_not_facts():
    # A user's KT (25 Oct): "Hi team, today I'm handing over our online learning
    # platform." was cited for "Online Learning Platform caches in Redis";
    # "Application ownership | Developers" and the handover check's "Not
    # covered during KT" lines were printed "no source"; a list of 16 names
    # cited three sentences.
    said = ["Hi team, today I'm handing over our online learning platform.",
            "The backend uses Python FastAPI and Redis for caching course information.",
            "Prometheus and Grafana monitor application performance.",
            "Developers own application code, while Platform Engineering manages Kubernetes."]
    sentences = transcript_sentences("", [{"start": float(i), "end": float(i + 1), "text": t} for i, t in enumerate(said)])
    ko = {"system_name": "Online Learning Platform", "rendered_sections": [
        {"section_id": "architecture_reference", "section_title": "Architecture Reference", "blocks": [
            {"type": "DecisionTable", "title": "Connections stated in the KT",
             "columns": ["From", "Connection", "To", "Said in the KT"],
             "rows": [{"From": "Online Learning Platform (Amazon EKS)", "Connection": "caches in", "To": "Redis",
                       "Said in the KT": said[1]}]},
            {"type": "NarrativeBlock", "title": "Architecture Knowledge", "paragraphs": ["FastAPI, Redis, Prometheus, Grafana"]}]},
        {"section_id": "ownership_escalation", "section_title": "Ownership & Escalation", "blocks": [
            {"type": "OwnershipTable", "title": "Ownership & Escalation",
             "rows": [{"role": "Application ownership", "team": "Developers"}]}]},
        {"section_id": "handover_completion", "section_title": "Handover Completion Check", "blocks": [
            {"type": "ChecklistBlock", "title": "Handover readiness", "items": [
                "Replacement can deploy safely: Not covered during KT"]}]}]}
    assert attach_evidence(ko, sentences)["unsourced"] == 0
    cited = {q for qs in dict(evidence_pairs(ko)).values() for q in qs}
    assert said[0] not in cited
    assert said[2] in cited and said[3] in cited
