"""P2-1: a KT built from several sessions. A follow-up session rebuilds the
document as the next version of the same KT; gaps it covers disappear, its
facts cite their session, and the KT keeps its workspace, owner and history."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sessions
from knowledge.evidence import transcript_sentences

SESSION_1 = (
    "Thanks for taking over Beacon, the parcel tracking service our couriers use. "
    "Beacon runs on Azure Kubernetes Service and stores events in Azure SQL. "
    "Azure DevOps builds the container and Helm deploys it; to roll back we redeploy the previous Helm release. "
    "Monitoring is Azure Monitor with Application Insights, and the key alert is the scan backlog. "
    "The worker loses its database connection after a failover; restart the worker pods and it drains. "
    "Never delete the Event Hubs consumer group, the billing team reads from it. "
    "The tracking squad owns Beacon, and escalation goes to Daniel Ruiz."
)
SESSION_2 = (
    "Picking up the disaster recovery questions from last time. "
    "Azure SQL has point in time restore for fourteen days and geo-replication to West Europe. "
    "Our RTO is four hours and the RPO is fifteen minutes. "
    "We tested a failover last spring and it took forty minutes."
)


def test_one_session_is_shaped_like_a_single_paste_and_several_are_tagged():
    one = [{"n": 1, "kind": "paste", "transcript": SESSION_1, "segments": None}]
    text, segs = sessions.combined_input(one)
    assert text == SESSION_1 and len(segs) == 1 and "session" not in segs[0] and segs[0]["untimed"]
    timed = {"n": 2, "kind": "audio", "transcript": "Backups run nightly.", "time_offset": 2.5,
             "segments": [{"start": 1.0, "end": 3.0, "text": "Backups run nightly."}]}
    text, segs = sessions.combined_input(one + [timed])
    assert [s["session"] for s in segs] == [1, 2] and segs[1]["start"] == 3.5 and segs[1]["end"] == 5.5
    sentences = transcript_sentences(text, segs)
    assert sentences[0]["start"] is None and sentences[0]["session"] == 1           # pasted: no time
    assert sentences[-1]["session"] == 2 and sentences[-1]["start"] == 3.5


@pytest.fixture
def kt(monkeypatch, tmp_path):
    """A finished one-session KT, run through the real pipeline without an LLM."""
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
    job_id = client.post("/kt-from-transcript", json={"transcript": SESSION_1}).json()["job_id"]
    Worker(worker_id="test-sessions").drain(job_id)
    job = pipeline.JOB_QUEUE[job_id]
    assert job["status"] in pipeline.COMPLETED_STATUSES
    job["created_by_id"] = "user-gina"
    pipeline.JOB_QUEUE[job_id] = job
    return client, pipeline, job_id, lambda: Worker(worker_id="test-sessions").drain(job_id)


def _dr_gaps(job):
    return [g for g in sessions.knowledge_gaps(job) if g.lower().startswith("disaster recovery")]


def test_a_second_session_closes_the_gaps_it_covers(kt):
    client, pipeline, job_id, drain = kt
    before = pipeline.JOB_QUEUE[job_id]
    agenda = client.get(f"/kt/{job_id}/sessions").json()["agenda"]
    assert agenda["next_session"] == 2 and _dr_gaps(before) and set(_dr_gaps(before)) <= set(agenda["topics"])

    resp = client.post(f"/kt/{job_id}/sessions", json={"transcript": SESSION_2})
    assert resp.status_code == 200 and resp.json()["pending"] is True
    assert client.post(f"/kt/{job_id}/sessions", json={"transcript": SESSION_2}).status_code == 409   # one at a time
    drain()

    after = pipeline.JOB_QUEUE[job_id]
    assert after["status"] in pipeline.COMPLETED_STATUSES and not after.get("session_pending")
    assert after["tenant_id"] == before["tenant_id"] and after["created_by_id"] == "user-gina"
    assert [s["n"] for s in after["sessions"]] == [1, 2] and after["document_version"] == 2
    closed = after["session_changes"]["closed_gaps"]
    assert closed and set(closed) <= set(sessions.knowledge_gaps(before))
    assert not set(closed) & set(sessions.knowledge_gaps(after))
    # Facts from both sessions, each source saying which.
    by_session = {s.get("session") for s in after["knowledge_object"]["sources"]}
    assert by_session == {1, 2}
    rto = [s for s in after["knowledge_object"]["sources"] if "RTO is four hours" in s["quote"]]
    assert rto and rto[0]["session"] == 2
    assert any("Built from 2 KT sessions" in n for n in after.get("notices") or [])
    # The one-session document stays in History.
    assert [v["version"] for v in pipeline.JOB_STORE.versions(job_id)] == [1, 2]
    status = client.get(f"/status/{job_id}").json()
    assert "segments" not in status and "transcript" not in status["sessions"][0]


def test_a_failed_rebuild_leaves_the_kt_as_it_was(kt, monkeypatch):
    client, pipeline, job_id, drain = kt
    before = pipeline.JOB_QUEUE[job_id]

    def broken(job_id, *a, **k):
        pipeline.JOB_QUEUE[job_id] = {"status": "failed", "error": "classifier crashed"}

    monkeypatch.setattr(pipeline, "run_kt_pipeline", broken)
    client.post(f"/kt/{job_id}/sessions", json={"transcript": SESSION_2})
    drain()
    after = pipeline.JOB_QUEUE[job_id]
    assert after["status"] == before["status"] and after["knowledge_object"] == before["knowledge_object"]
    assert "classifier crashed" in after["session_error"] and not after.get("session_pending")


def test_a_signed_kt_takes_no_more_sessions(kt):
    client, pipeline, job_id, _ = kt
    job = pipeline.JOB_QUEUE[job_id]
    job["signoff"] = {"status": "signed", "signed_version": 1}
    pipeline.JOB_QUEUE[job_id] = job
    resp = client.post(f"/kt/{job_id}/sessions", json={"transcript": SESSION_2})
    assert resp.status_code == 409 and "signed" in resp.json()["detail"]
