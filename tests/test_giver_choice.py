"""Choosing who is handing over in a conversation KT (speaker_roles,
POST /kt/{job_id}/giver): the giver is detected, shown with the KT, and the
user can pick another speaker; the document is rebuilt as a new version
with what that person said as the facts."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sessions  # noqa: E402

CONVERSATION = """Arjun: This is Ledgerline, our invoicing service on Amazon EKS. Invoices are stored in PostgreSQL on Amazon RDS. Backups are kept for 14 days.
Meera: So the reconciliation job runs nightly?
Arjun: Yes, every night at two, and it must never run twice in one day.
Tom: Also, Meera, you'll own the on-call rotation from next month."""


def test_a_rebuild_keeps_who_said_each_turn():
    text, segs = sessions.combined_input([{"n": 1, "kind": "paste", "transcript": CONVERSATION, "segments": None}])
    assert [s["speaker"] for s in segs] == ["Arjun", "Meera", "Arjun", "Tom"]


@pytest.fixture
def kt(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import ai
    import main
    import pipeline
    from context_mapper import ContextMappingPipeline
    from job_queue import TaskQueue
    from job_store import JobStore, PersistentJobs
    from kt_schema_loader import SCHEMA
    from worker import Worker

    store = JobStore(str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setattr(pipeline, "TASKS", TaskQueue(store))
    monkeypatch.setattr(pipeline, "get_llm_provider", lambda: None)
    monkeypatch.setattr(ai, "get_llm_provider", lambda: None)
    monkeypatch.setattr(pipeline, "record_candidates", lambda *a, **k: None)
    if pipeline.MAPPER_PIPELINE is None:
        monkeypatch.setattr(pipeline, "MAPPER_PIPELINE", ContextMappingPipeline(SCHEMA, llm_fallback_fn=None))
    client = TestClient(main.app)
    job_id = client.post("/kt-from-transcript", json={"transcript": CONVERSATION}).json()["job_id"]
    Worker(worker_id="test-giver").drain(job_id)
    assert pipeline.JOB_QUEUE[job_id]["status"] in pipeline.COMPLETED_STATUSES
    return client, pipeline, job_id, lambda: Worker(worker_id="test-giver").drain(job_id)


def test_the_giver_is_detected_and_can_be_chosen(kt):
    client, pipeline, job_id, drain = kt
    status = client.get(f"/status/{job_id}").json()
    roles = status["speaker_roles"]
    assert (roles["giver"], roles["receivers"], roles["detected"]) == ("Arjun", ["Meera"], True)
    assert any(n.startswith("Handed over by Arjun (detected") for n in status["notices"])
    # Sources say who said each sentence.
    speakers = {s.get("speaker") for s in status["knowledge_object"]["sources"]}
    assert {"Arjun", "Tom"} <= speakers

    assert client.post(f"/kt/{job_id}/giver", json={"giver": "Nobody"}).status_code == 400
    resp = client.post(f"/kt/{job_id}/giver", json={"giver": "Tom"})
    assert resp.status_code == 200 and resp.json()["giver"] == "Tom"
    assert client.post(f"/kt/{job_id}/giver", json={"giver": "Arjun"}).status_code == 409    # one at a time
    drain()
    status = client.get(f"/status/{job_id}").json()
    assert not status.get("giver_pending") and not status.get("giver_error")
    assert status["giver"] == "Tom" and status["speaker_roles"]["giver"] == "Tom"
    assert not status["speaker_roles"]["detected"]
    assert status["document_version"] == 2
    assert any(n.startswith("Handed over by Tom (chosen") for n in status["notices"])

    # Back to detecting it.
    client.post(f"/kt/{job_id}/giver", json={"giver": None})
    drain()
    status = client.get(f"/status/{job_id}").json()
    assert status["giver"] is None and status["speaker_roles"]["giver"] == "Arjun"
