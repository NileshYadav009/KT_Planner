"""P3-1: a question about a KT is answered with the document's own
statements, each with its section and source sentence. Nothing is written
by a model; a topic the KT never covered is said to be a gap."""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ask
from renderers.blocks.common import NOT_COVERED_MESSAGE


def _document():
    return {"system_name": "Quarry", "rendered_sections": [
        {"section_id": "system_overview", "section_title": "System Overview", "blocks": [
            {"type": "TechnologyGrid", "title": "At a glance",
             "rows": [{"label": "Business criticality", "value": "High"}]},
            {"type": "NarrativeBlock", "title": "Overview", "sources": [[1]],
             "paragraphs": ["Quarry is the quoting engine the sales team uses; the back end is Java on RHEL."]}]},
        {"section_id": "ownership_escalation", "section_title": "Ownership & Escalation", "blocks": [
            {"type": "NarrativeBlock", "title": "Escalation", "sources": [[2]],
             "paragraphs": ["If you are stuck, Marta Kowalski is the person to call."]}]},
        {"section_id": "deployment_and_rollback", "section_title": "Deployment & Rollback", "blocks": [
            {"type": "NarrativeBlock", "title": "Rollback", "sources": [[3]],
             "paragraphs": ["To undo a release, run the same playbook with the previous version number."]}]},
        {"section_id": "disaster_recovery", "section_title": "Disaster Recovery", "blocks": [
            {"type": "NarrativeBlock", "title": "Backups", "sources": [[4]],
             "paragraphs": ["Veeam backs up the VMs every night."]}]},
        {"section_id": "cost_optimization", "section_title": "Cost Optimization", "blocks": [
            {"type": "NarrativeBlock", "title": "Cost", "paragraphs": [NOT_COVERED_MESSAGE]}]},
        {"section_id": "tribal_knowledge", "section_title": "Additional Notes", "blocks": [
            {"type": "ChecklistBlock", "title": "Notes",
             "items": ["If you are stuck, Marta Kowalski is the person to call… Safety-critical"]}]},
        {"section_id": "kt_coverage", "section_title": "KT Coverage", "blocks": [
            {"type": "ChecklistBlock", "title": "Gaps", "items": ["Cost Optimization was not covered"]}]},
    ], "sections": [{"id": "kt_coverage", "_knowledge_gaps": ["Cost Optimization was not covered in the session"]}],
        "sources": [{"id": 1, "quote": "Quarry is the quoting engine.", "start": 3.0},
                    {"id": 2, "quote": "Marta Kowalski is the person to call.", "start": 410.0, "speaker": "Speaker 1"},
                    {"id": 3, "quote": "run the same playbook with the previous version number", "start": 95.0},
                    {"id": 4, "quote": "Veeam backs up the VMs every night.", "start": 300.0}]}


def test_a_who_question_is_answered_by_the_escalation_statement_with_its_source():
    r = ask.answer(_document(), "Who do I call when quoting is down?")
    top = r["answers"][0]
    assert top["section"] == "Ownership & Escalation" and "Marta Kowalski" in top["text"]
    assert top["sources"] == [{"id": 2, "quote": "Marta Kowalski is the person to call.", "start": 410.0,
                               "speaker": "Speaker 1", "session": None}]
    # The Additional Notes digest of the same line is not a second answer.
    assert sum("Marta" in a["text"] for a in r["answers"]) == 1


def test_roll_back_is_not_the_back_end():
    r = ask.answer(_document(), "How do I roll back a bad release?")
    assert r["answers"][0]["section_id"] == "deployment_and_rollback"
    assert all(a["section_id"] != "system_overview" for a in r["answers"])


def test_a_term_the_kt_never_mentions_is_named():
    r = ask.answer(_document(), "What is the RTO?")
    assert r["topics"] == ["disaster_recovery"] and r["never_mentions"] == ["RTO"]
    assert r["answers"][0]["text"] == "Veeam backs up the VMs every night."


def test_an_uncovered_topic_is_reported_as_a_gap_not_answered_with_noise():
    r = ask.answer(_document(), "How much does it cost to run?")
    assert r["answers"] == [] or all(not a["on_topic"] for a in r["answers"])
    assert r["not_covered"] == ["Cost Optimization was not covered in the session"]


def test_an_unrelated_question_gets_nothing_and_a_table_value_keeps_its_label():
    assert ask.answer(_document(), "What is the weather like?")["answers"] == []
    labelled = ask.answer(_document(), "How critical is it?")["answers"]
    assert labelled and labelled[0]["text"] == "Business criticality: High"


def test_the_route_answers_for_any_role_in_the_workspace_only(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import auth
    import main
    import pipeline
    from job_store import JobStore, PersistentJobs

    store = JobStore(str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setenv("CONTINUUM_AUTH", "required")
    tenant = auth.create_tenant("Ask Co " + uuid.uuid4().hex[:6])
    reader = auth.create_key(tenant, "reader", role="receiver")
    outsider = auth.create_key(auth.create_tenant("Elsewhere"), "x", role="admin")
    pipeline.JOB_QUEUE["ask-1"] = {"status": "completed", "tenant_id": tenant, "knowledge_object": _document()}
    client = TestClient(main.app)

    resp = client.post("/kt/ask-1/ask", json={"question": "Who do I call?"}, headers={"X-API-Key": reader})
    assert resp.status_code == 200 and "Marta" in resp.json()["answers"][0]["text"]
    assert client.post("/kt/ask-1/ask", json={"question": " "}, headers={"X-API-Key": reader}).status_code == 400
    assert client.post("/kt/ask-1/ask", json={"question": "Who?"}, headers={"X-API-Key": outsider}).status_code == 404
