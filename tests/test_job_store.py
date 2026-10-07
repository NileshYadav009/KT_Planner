"""P0-2: jobs and documents survive a restart. Before, JOB_QUEUE was a plain
dict: a deploy erased every KT and the recording was already deleted."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from job_store import INTERRUPTED_MESSAGE, JobStore, PersistentJobs


@pytest.fixture
def store(tmp_path):
    return JobStore(str(tmp_path / "jobs.sqlite"))


def test_a_job_written_before_a_restart_is_readable_after_it(store):
    jobs = PersistentJobs(store)
    jobs["job-1"] = {"status": "processing", "progress": 0, "tenant_id": "acme"}
    jobs["job-1"] = {"status": "completed", "transcript": "x" * 50_000,
                     "knowledge_object": {"system_name": "TripWise", "rendered_sections": [{"section_id": "s"}]}}
    after_restart = PersistentJobs(JobStore(store.path))     # new process, same database
    job = after_restart["job-1"]
    assert job["status"] == "completed"
    assert job["knowledge_object"]["system_name"] == "TripWise"
    assert job["tenant_id"] == "acme"                         # kept from the first write
    assert store.list("acme")[0]["title"] == "TripWise"


def test_jobs_running_at_shutdown_are_marked_interrupted(store):
    PersistentJobs(store)["job-2"] = {"status": "processing", "progress": 30}
    assert store.mark_interrupted() == 1
    job = store.get("job-2")
    assert job["status"] == "failed" and job["error"] == INTERRUPTED_MESSAGE


def test_unknown_jobs_are_missing_and_deletes_are_permanent(store):
    jobs = PersistentJobs(store)
    assert "nope" not in jobs
    jobs["job-3"] = {"status": "completed"}
    del jobs["job-3"]
    assert store.get("job-3") is None


def test_running_jobs_are_never_evicted_from_the_working_set(store):
    jobs = PersistentJobs(store, cache_size=2)
    jobs["running"] = {"status": "processing", "progress": 40}
    for i in range(5):
        jobs[f"done-{i}"] = {"status": "completed"}
    jobs["running"]["progress"] = 80                          # live progress update
    assert jobs["running"]["progress"] == 80


def test_export_works_after_a_simulated_restart(monkeypatch):
    pytest.importorskip("weasyprint")
    from fastapi.testclient import TestClient
    import main
    import pipeline

    pipeline.JOB_QUEUE["restart-0001"] = {
        "status": "completed", "coverage": {},
        "knowledge_object": {"system_name": "T", "rendered_sections": [
            {"section_id": "s", "section_title": "S", "blocks": [{"type": "NarrativeBlock", "title": "S", "paragraphs": ["x"]}]}]},
    }
    # A restarted process: empty working set over the same database.
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(pipeline.JOB_STORE))
    client = TestClient(main.app)
    assert client.get("/export/pdf/restart-0001").status_code == 200
    assert any(j["job_id"] == "restart-0001" for j in client.get("/jobs").json()["jobs"])
    pipeline.JOB_STORE.delete("restart-0001")


def test_feedback_corrections_are_persisted():
    from fastapi.testclient import TestClient
    import main
    import pipeline

    pipeline.JOB_QUEUE["fb-0001"] = {"status": "completed", "coverage": {
        "monitoring_observability": {"title": "M", "content": ["Datadog."], "sentences": [{"text": "Datadog."}]}}}
    TestClient(main.app).post("/feedback", json={"job_id": "fb-0001", "sentence_id": 0,
                                                 "corrected_classification": "danger_zones"})
    stored = pipeline.JOB_STORE.get("fb-0001")
    assert stored["human_feedback"][0]["corrected_classification"] == "danger_zones"
    pipeline.JOB_STORE.delete("fb-0001")


def test_a_database_with_the_first_stored_pdf_table_still_exports(tmp_path):
    """The stored-PDF table first had a rendered_at column; a database from
    then made every download fail with "no such column: job_version"."""
    import sqlite3

    path = str(tmp_path / "old.sqlite")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE documents (job_id TEXT PRIMARY KEY, pdf BLOB NOT NULL, rendered_at REAL NOT NULL)")
        db.execute("INSERT INTO documents VALUES ('old-0001', x'25504446', 1.0)")
    store = JobStore(path)
    version = store.put("old-0001", {"status": "completed"})
    assert store.current_document("old-0001") is None             # the old cache is gone, not misread
    store.save_document("old-0001", b"%PDF new", version)
    assert store.current_document("old-0001") == b"%PDF new"
