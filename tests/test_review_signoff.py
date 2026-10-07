"""P1-1: reviewers edit the server's document; each save is a version, an
edit appears in the next PDF, earlier versions can be read and exported.
P1-3: a named giver, receiver and approver acknowledge a version; it cannot
be signed with an open knowledge gap; the signed version is locked and the
signature is in the audit log."""
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import auth
from job_queue import TaskQueue
from job_store import JobStore, PersistentJobs
from renderers.blocks.common import NOT_COVERED_MESSAGE
from renderers.sections.kt_coverage import KNOWLEDGE_GAPS_TITLE

GAPS = ["Disaster recovery was not covered in the session", "Cost controls were not covered in the session"]


def _document():
    return {"system_name": "TripWise", "rendered_sections": [
        {"section_id": "monitoring_observability", "section_title": "Monitoring", "blocks": [
            {"type": "NarrativeBlock", "title": "Monitoring", "sources": [[1], [2]],
             "paragraphs": ["Datadog monitors the booking API.", "PagerDuty pages the on-call engineer."]}]},
        {"section_id": "danger_zones", "section_title": "Danger Zones", "blocks": [
            {"type": "WarningBlock", "title": "Danger", "warnings": ["Never delete bookings by hand."], "sources": [[3]]}]},
        {"section_id": "ownership_escalation", "section_title": "Ownership", "blocks": [
            {"type": "OwnershipTable", "title": "Owners", "rows": [{"role": "Bookings", "team": "Platform team"}]}]},
        {"section_id": "security_controls", "section_title": "Security", "blocks": [
            {"type": "NarrativeBlock", "title": "Security", "paragraphs": [NOT_COVERED_MESSAGE]}]},
        {"section_id": "kt_coverage", "section_title": "KT Coverage", "blocks": [
            {"type": "ChecklistBlock", "title": KNOWLEDGE_GAPS_TITLE, "items": list(GAPS)}]},
        {"section_id": "signoff", "section_title": "Sign-off", "blocks": [
            {"type": "NarrativeBlock", "title": "Sign-off", "paragraphs": [NOT_COVERED_MESSAGE]}]},
    ], "sources": [{"id": 1, "quote": "Datadog monitors the booking API.", "start": 12.0, "end": 15.0},
                   {"id": 2, "quote": "PagerDuty pages the on-call engineer.", "start": 15.0, "end": 18.0},
                   {"id": 3, "quote": "Never delete bookings by hand.", "start": 40.0, "end": 42.0}]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    import pdf_rendering
    import pipeline

    store = JobStore(str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setattr(pipeline, "TASKS", TaskQueue(store))
    printed = []

    def fake_pdf(job_id, job, created=None):
        import signoff

        sections = signoff.apply_to_document(job["knowledge_object"]["rendered_sections"], job,
                                             int(job.get("document_version") or 1))
        printed.append(sections)
        return b"%PDF fake"

    monkeypatch.setattr(pipeline, "build_job_pdf", fake_pdf)
    monkeypatch.setattr(pdf_rendering, "build_job_pdf", fake_pdf)
    pipeline.printed = printed
    return pipeline


@pytest.fixture
def team(env, monkeypatch):
    from fastapi.testclient import TestClient
    import main

    monkeypatch.setenv("CONTINUUM_AUTH", "required")
    tenant = auth.create_tenant("Review Co " + uuid.uuid4().hex[:6])

    def person(name, role):
        email = f"{name}@{tenant}.example"
        user_id = auth.add_user(tenant, email, role)
        token = auth.create_session(auth.Principal(tenant_id=tenant, role=role, key_label=email, user_id=user_id,
                                                   method="sso"))
        return {"email": email, "h": {"Cookie": f"{auth.SESSION_COOKIE}={token}", "X-Continuum-CSRF": "1"}}

    people = {"giver": person("gina", "giver"), "receiver": person("rita", "receiver"),
              "approver": person("arun", "reviewer"), "outsider": person("otto", "giver")}
    job_id = "review-" + uuid.uuid4().hex[:8]
    env.JOB_QUEUE[job_id] = {"status": "completed", "tenant_id": tenant, "knowledge_object": _document(),
                             "coverage": {"monitoring_observability": {"title": "Monitoring", "content": [
                                 "PagerDuty pages the on-call engineer."], "sentences": [
                                 {"text": "PagerDuty pages the on-call engineer.",
                                  "assigned_sections": ["monitoring_observability"]}]}}}
    client = TestClient(main.app)
    return client, job_id, tenant, people


def _units(job_or_ko, section_id):
    ko = job_or_ko.get("knowledge_object", job_or_ko)
    section = next(s for s in ko["rendered_sections"] if s["section_id"] == section_id)
    out = []
    for block in section["blocks"]:
        for key in ("paragraphs", "items", "warnings"):
            out.extend(block.get(key) or [])
        out.extend(f"{r.get('role')}|{r.get('team')}" for r in block.get("rows") or [] if "role" in r)
    return out


def _edit(client, job_id, who, base, *edits):
    return client.post(f"/documents/{job_id}/edits", json={"base_version": base, "edits": list(edits)}, headers=who["h"])


# --------------------------------------------------------------------------
# P1-1 Reviewer edits and versions
# --------------------------------------------------------------------------

def test_an_edit_is_a_new_version_and_reaches_the_next_pdf(team, env):
    client, job_id, _, people = team
    resp = _edit(client, job_id, people["giver"], 1, {"op": "replace", "section": "monitoring_observability",
                                                     "block": 0, "unit": 0, "text": "Datadog monitors the API and the queue."})
    assert resp.status_code == 200, resp.text
    assert resp.json()["version"] == 2

    assert client.get(f"/export/pdf/{job_id}", headers=people["giver"]["h"]).status_code == 200
    assert "Datadog monitors the API and the queue." in _units({"rendered_sections": env.printed[-1]},
                                                              "monitoring_observability")
    # Version 1 is still there, as generated.
    v1 = client.get(f"/documents/{job_id}/versions/1", headers=people["receiver"]["h"]).json()
    assert _units(v1, "monitoring_observability")[0] == "Datadog monitors the booking API."
    assert client.get(f"/export/pdf/{job_id}?version=1", headers=people["receiver"]["h"]).status_code == 200
    assert "Datadog monitors the booking API." in _units({"rendered_sections": env.printed[-1]},
                                                        "monitoring_observability")
    history = client.get(f"/documents/{job_id}/versions", headers=people["receiver"]["h"]).json()
    assert history["current"] == 2 and [v["version"] for v in history["versions"]] == [1, 2]
    assert history["versions"][1]["summary"] == "1 edited"


def test_a_save_made_against_an_old_version_is_refused(team):
    client, job_id, _, people = team
    edit = {"op": "approve", "section": "danger_zones", "block": 0, "unit": 0}
    assert _edit(client, job_id, people["giver"], 1, edit).status_code == 200
    stale = _edit(client, job_id, people["approver"], 1, edit)
    assert stale.status_code == 409 and "changed since you opened it" in stale.json()["detail"]


def test_approve_delete_move_and_add_in_one_save(team, env):
    client, job_id, _, people = team
    resp = _edit(client, job_id, people["approver"], 1,
                 {"op": "move", "section": "monitoring_observability", "block": 0, "unit": 1, "to_section": "danger_zones"},
                 {"op": "approve", "section": "ownership_escalation", "block": 0, "unit": 0},
                 {"op": "replace", "section": "ownership_escalation", "block": 0, "unit": 0, "cells": {"team": "SRE team"}},
                 {"op": "add", "section": "security_controls", "text": "MFA is required for console access."},
                 {"op": "delete", "section": "danger_zones", "block": 0, "unit": 0})
    assert resp.status_code == 200, resp.text
    job = env.JOB_QUEUE[job_id]
    assert _units(job, "monitoring_observability") == ["Datadog monitors the booking API."]
    assert _units(job, "danger_zones") == ["PagerDuty pages the on-call engineer."]       # moved, not copied
    moved_block = next(s for s in job["knowledge_object"]["rendered_sections"] if s["section_id"] == "danger_zones")["blocks"][0]
    assert moved_block["sources"] == [[2]] and moved_block["review"][0]["state"] == "moved"
    assert _units(job, "security_controls") == ["MFA is required for console access."]   # no "not covered" left
    assert _units(job, "ownership_escalation") == ["Bookings|SRE team"]
    assert resp.json()["summary"] == "1 approved, 1 edited, 1 deleted, 1 added, 1 moved"


def test_a_receiver_cannot_edit_and_bad_edits_change_nothing(team, env):
    client, job_id, _, people = team
    edit = {"op": "replace", "section": "danger_zones", "block": 0, "unit": 0, "text": "x"}
    assert _edit(client, job_id, people["receiver"], 1, edit).status_code == 403
    bad = _edit(client, job_id, people["giver"], 1, edit,
                {"op": "replace", "section": "danger_zones", "block": 0, "unit": 9, "text": "y"})
    assert bad.status_code == 400
    assert _units(env.JOB_QUEUE[job_id], "danger_zones") == ["Never delete bookings by hand."]
    assert env.JOB_QUEUE[job_id].get("document_version") is None


def test_a_section_correction_moves_the_sentence_and_changes_the_pdf(team, env):
    client, job_id, _, people = team
    resp = client.post("/feedback", json={"job_id": job_id, "sentence_id": "PagerDuty pages the on-call engineer.",
                                          "corrected_classification": "ownership_escalation"}, headers=people["giver"]["h"])
    assert resp.status_code == 200 and resp.json()["document_version"] == 2
    job = env.JOB_QUEUE[job_id]
    assert "PagerDuty pages the on-call engineer." not in _units(job, "monitoring_observability")
    assert "PagerDuty pages the on-call engineer." in _units(job, "ownership_escalation")


# --------------------------------------------------------------------------
# P1-3 Sign-off
# --------------------------------------------------------------------------

def _start(client, job_id, people):
    return client.post(f"/signoff/{job_id}/start", headers=people["giver"]["h"], json={
        r: people[r]["email"] for r in ("giver", "receiver", "approver")})


def _ack(client, job_id, people, role, version):
    return client.post(f"/signoff/{job_id}/acknowledge", headers=people[role]["h"], json={"role": role, "version": version})


def test_a_kt_cannot_be_signed_with_an_open_gap(team):
    client, job_id, _, people = team
    state = _start(client, job_id, people).json()
    assert state["status"] == "in_review" and [g["text"] for g in state["gaps"]] == GAPS
    blocked = _ack(client, job_id, people, "receiver", 1)
    assert blocked.status_code == 409 and "2 knowledge gap(s) are still open" in blocked.json()["detail"]


def test_signing_needs_all_three_on_one_version_then_locks_it(team, env):
    client, job_id, tenant, people = team
    gaps = _start(client, job_id, people).json()["gaps"]
    gap_url = f"/signoff/{job_id}/gaps/"
    assert client.post(gap_url + gaps[0]["id"], headers=people["outsider"]["h"],
                       json={"resolution": "closed", "note": "x"}).status_code == 403
    assert client.post(gap_url + gaps[0]["id"], headers=people["receiver"]["h"],
                       json={"resolution": "closed"}).status_code == 400                 # a note is required
    client.post(gap_url + gaps[0]["id"], headers=people["receiver"]["h"],
                json={"resolution": "closed", "note": "DR runbook walked through on 7 Oct"})
    client.post(gap_url + gaps[1]["id"], headers=people["approver"]["h"],
                json={"resolution": "accepted_risk", "note": "Cost review is next quarter"})

    assert _ack(client, job_id, people, "approver", 1).status_code == 200
    # Someone else cannot acknowledge for the giver.
    assert client.post(f"/signoff/{job_id}/acknowledge", headers=people["receiver"]["h"],
                       json={"role": "giver", "version": 1}).status_code == 403
    # An edit after an acknowledgement clears it: people acknowledge what the document says now.
    assert _edit(client, job_id, people["giver"], 1, {"op": "approve", "section": "danger_zones", "block": 0,
                                                     "unit": 0}).status_code == 200
    state = client.get(f"/signoff/{job_id}", headers=people["approver"]["h"]).json()
    assert state["acknowledgements"] == {} and state["current_version"] == 2
    assert _ack(client, job_id, people, "approver", 1).status_code == 409                # not the current version

    for role in ("giver", "receiver"):
        assert _ack(client, job_id, people, role, 2).json()["status"] == "in_review"
    signed = _ack(client, job_id, people, "approver", 2).json()
    assert signed["status"] == "signed" and signed["signed_version"] == 2

    # Locked.
    locked = _edit(client, job_id, people["giver"], 2, {"op": "approve", "section": "danger_zones", "block": 0, "unit": 0})
    assert locked.status_code == 409 and "signed" in locked.json()["detail"]
    # The document's Sign-off section is the record, in the UI and the PDF.
    status = client.get(f"/status/{job_id}", headers=people["receiver"]["h"]).json()
    signoff_section = next(s for s in status["knowledge_object"]["rendered_sections"] if s["section_id"] == "signoff")
    assert "Signed: version 2" in signoff_section["blocks"][0]["paragraphs"][0]
    client.get(f"/export/pdf/{job_id}", headers=people["receiver"]["h"])
    printed = next(s for s in env.printed[-1] if s["section_id"] == "signoff")
    assert printed["blocks"][2]["rows"][1]["Resolution"] == "Accepted as a risk"
    # The signature is in the audit log.
    events = {e["action"]: e["detail"] for e in auth.audit_events(tenant)}
    assert events["kt_signed"]["version"] == 2 and set(events["kt_signed"]["participants"]) == {
        "giver", "receiver", "approver"}
    assert {g["resolution"] for g in events["kt_signed"]["gaps"]} == {"closed", "accepted_risk"}


def test_participants_must_be_three_people_of_the_workspace(team):
    client, job_id, _, people = team
    same = client.post(f"/signoff/{job_id}/start", headers=people["giver"]["h"], json={
        "giver": people["giver"]["email"], "receiver": people["giver"]["email"], "approver": people["approver"]["email"]})
    assert same.status_code == 400
    stranger = client.post(f"/signoff/{job_id}/start", headers=people["giver"]["h"], json={
        "giver": people["giver"]["email"], "receiver": "nobody@elsewhere.example", "approver": people["approver"]["email"]})
    assert stranger.status_code == 400 and "not an active person" in stranger.json()["detail"]
