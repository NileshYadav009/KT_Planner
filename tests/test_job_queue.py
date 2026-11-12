"""P1-4: KTs run from a persistent queue on workers with a concurrency limit,
not as BackgroundTasks inside the API process. A double submit with the same
Idempotency-Key is one KT, the PDF is rendered by the worker and stored, and
a KT survives its worker dying."""
import os
import shutil
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import job_queue
from job_queue import GAVE_UP_MESSAGE, TaskQueue
from job_store import INTERRUPTED_MESSAGE, JobStore, PersistentJobs
from worker import Worker

TRANSCRIPT = "The payments API runs on EKS and alerts go to PagerDuty."


@pytest.fixture
def env(tmp_path, monkeypatch):
    """pipeline with its own job database, so queue tests see only their tasks."""
    import pipeline

    store = JobStore(str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setattr(pipeline, "TASKS", TaskQueue(store))
    return pipeline


def _complete(pipeline, job_id):
    pipeline.JOB_QUEUE[job_id] = {
        "status": "completed", "coverage": {},
        "knowledge_object": {"system_name": "Payments", "rendered_sections": [
            {"section_id": "s", "section_title": "S", "blocks": [{"type": "NarrativeBlock", "title": "S", "paragraphs": ["x"]}]}]},
    }


@pytest.fixture
def client(env, monkeypatch):
    from fastapi.testclient import TestClient
    import main

    ran, renders = [], []

    def fake_pipeline(job_id, transcript, segments=None, screen=None, warnings=None):
        ran.append(job_id)
        _complete(env, job_id)

    monkeypatch.setattr(env, "run_kt_pipeline", fake_pipeline)
    monkeypatch.setattr(env, "build_job_pdf", lambda job_id, job, created=None: renders.append(job_id) or b"%PDF-1.7 stored")
    c = TestClient(main.app)          # no startup: no embedded worker, tests run the worker themselves
    c.ran, c.renders = ran, renders
    return c


def _submit(client, key=None, transcript=TRANSCRIPT):
    headers = {"Idempotency-Key": key} if key else {}
    return client.post("/kt-from-transcript", json={"transcript": transcript, "force": True}, headers=headers)


# --------------------------------------------------------------------------
# Submitting
# --------------------------------------------------------------------------

def test_a_submission_is_queued_not_run_in_the_request(client, env):
    first, second = _submit(client).json(), _submit(client).json()
    assert first["status"] == "queued" and client.ran == []
    status = client.get(f"/status/{second['job_id']}").json()
    assert status["status"] == "queued" and status["queue_position"] == 1     # one KT ahead of it
    assert env.TASKS.task(first["job_id"])["state"] == "queued"


def test_a_worker_runs_it_stores_the_pdf_and_forgets_the_transcript(client, env):
    job_id = _submit(client).json()["job_id"]
    assert Worker(worker_id="w1").drain(job_id) == 1
    assert client.ran == [job_id]
    assert client.get(f"/status/{job_id}").json()["status"] == "completed"
    task = env.TASKS.task(job_id)
    assert task["state"] == "done" and task["payload"] == {}             # the queued transcript is gone
    assert client.renders == [job_id]                                     # rendered by the worker

    for _ in range(2):
        resp = client.get(f"/export/pdf/{job_id}")
        assert resp.status_code == 200 and resp.content == b"%PDF-1.7 stored"
    assert client.renders == [job_id]                                     # downloads are reads


def test_a_changed_kt_is_rendered_again_on_the_next_download(client, env):
    job_id = _submit(client).json()["job_id"]
    Worker(worker_id="w1").drain(job_id)
    job = env.JOB_QUEUE[job_id]
    job["human_feedback"] = [{"sentence_id": 0}]
    env.JOB_QUEUE[job_id] = job                                          # a correction
    assert client.get(f"/export/pdf/{job_id}").status_code == 200
    assert client.renders == [job_id, job_id]
    client.get(f"/export/pdf/{job_id}")
    assert client.renders == [job_id, job_id]                            # and stored again


def test_a_failure_outside_the_pipeline_fails_the_kt_with_a_message(client, env, monkeypatch):
    monkeypatch.setattr(env, "run_kt_pipeline", lambda *a, **k: (_ for _ in ()).throw(MemoryError("boom")))
    job_id = _submit(client).json()["job_id"]
    Worker(worker_id="w1").drain(job_id)
    job = client.get(f"/status/{job_id}").json()
    assert job["status"] == "failed" and "Submit the KT again" in job["error"]
    assert env.TASKS.task(job_id)["state"] == "failed"


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------

def test_the_same_idempotency_key_is_one_kt(client, env):
    first = _submit(client, key="kt-7f3a").json()
    again = _submit(client, key="kt-7f3a").json()
    assert again["job_id"] == first["job_id"] and again["duplicate"] is True
    other = _submit(client, key="kt-other").json()
    assert other["job_id"] != first["job_id"]
    assert Worker(worker_id="w1").drain() == 2                           # two KTs, not three
    assert _submit(client, key="kt-7f3a").json()["status"] == "completed"   # a late retry sees the result


def test_idempotency_keys_are_per_workspace_and_validated(client, env):
    assert env.TASKS.reserve_key("acme", "k1", "job-a") == "job-a"
    assert env.TASKS.reserve_key("globex", "k1", "job-b") == "job-b"
    assert env.TASKS.reserve_key("acme", "k1", "job-c") == "job-a"
    assert _submit(client, key="has spaces").status_code == 400
    assert _submit(client, key="x" * 129).status_code == 400


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="needs ffprobe")
def test_a_refused_upload_frees_its_key(client, env, tmp_path):
    notes = tmp_path / "notes.mp3"
    notes.write_text("not a recording")
    with open(notes, "rb") as fh:
        resp = client.post("/upload", files={"file": ("notes.mp3", fh, "audio/mpeg")}, headers={"Idempotency-Key": "up-1"})
    assert resp.status_code == 422
    assert env.TASKS.reserve_key("local", "up-1", "job-next") == "job-next"   # a corrected retry is a new KT


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------

def _queue(env, n, kind="transcript"):
    ids = []
    for i in range(n):
        job_id = f"queue-job-{i:04d}"
        env.JOB_QUEUE[job_id] = {"status": "queued", "progress": 0, "tenant_id": "local"}
        env.TASKS.enqueue(job_id, kind, {"transcript": TRANSCRIPT})
        ids.append(job_id)
    return ids


def test_tasks_are_claimed_in_order_and_only_once(env):
    a, b = _queue(env, 2)
    assert env.TASKS.claim("w1")["job_id"] == a
    assert env.TASKS.claim("w2")["job_id"] == b
    assert env.TASKS.claim("w3") is None


def test_a_kt_whose_worker_died_runs_again_then_fails_with_a_reason(env, monkeypatch):
    job_id, = _queue(env, 1)
    monkeypatch.setattr(job_queue, "LEASE_SECONDS", -1)                    # every lease has already run out
    assert env.TASKS.claim("w1")["attempt"] == 1                           # w1 dies (OOM, restart)
    assert env.TASKS.claim("w2")["attempt"] == 2                           # picked up again; w2 dies too
    assert env.TASKS.claim("w3") is None
    job = env.JOB_QUEUE[job_id]
    assert job["status"] == "failed" and job["error"] == GAVE_UP_MESSAGE.format(attempts=2)


def test_giving_up_on_one_kt_does_not_hold_back_the_next(env, monkeypatch):
    # claim() reused `job_id` for the KT it gave up, then looked only for
    # that KT: the next queued one waited for another poll.
    first, second = _queue(env, 2)
    monkeypatch.setattr(job_queue, "LEASE_SECONDS", -1)
    assert [env.TASKS.claim("w")["job_id"] for _ in range(2)] == [first, first]
    claimed = env.TASKS.claim("w")
    assert claimed is not None and claimed["job_id"] == second
    assert env.JOB_QUEUE[first]["status"] == "failed"


def _give_up(env, monkeypatch):
    """Every queued KT's worker dies on every attempt, until the queue gives up."""
    monkeypatch.setattr(job_queue, "LEASE_SECONDS", -1)                    # every lease has already run out
    while env.TASKS.claim("dying-worker") is not None:
        pass
    monkeypatch.setattr(job_queue, "LEASE_SECONDS", 120.0)


def test_a_kt_whose_worker_died_is_tried_again_from_its_kept_input(client, env, monkeypatch):
    # The due-diligence review (P1-6): the user had to submit it again, a
    # recording uploaded again.
    job_id = _submit(client).json()["job_id"]
    _give_up(env, monkeypatch)
    status = client.get(f"/status/{job_id}").json()
    assert status["status"] == "failed" and status["can_retry"] is True and "Try again" in status["error"]
    assert client.post(f"/jobs/{job_id}/retry").json() == {"status": "queued", "job_id": job_id}
    assert env.JOB_QUEUE[job_id]["status"] == "queued" and env.JOB_QUEUE[job_id]["error"] is None
    Worker(worker_id="w4").drain(job_id)
    assert env.JOB_QUEUE[job_id]["status"] == "completed" and client.ran == [job_id]
    assert env.TASKS.task(job_id)["payload"] == {}                         # forgotten once it ran
    assert client.post(f"/jobs/{job_id}/retry").status_code == 409


def test_a_kt_that_failed_on_its_input_or_in_the_pipeline_is_not_offered_a_retry(client, env, monkeypatch):
    def broken(job_id, transcript, **kwargs):
        raise RuntimeError("a bug in a stage")
    monkeypatch.setattr(env, "run_kt_pipeline", broken)
    job_id = _submit(client).json()["job_id"]
    Worker(worker_id="w1").drain(job_id)
    status = client.get(f"/status/{job_id}").json()
    assert status["status"] == "failed" and status["can_retry"] is False
    assert client.post(f"/jobs/{job_id}/retry").status_code == 409


def test_a_kept_upload_is_deleted_after_a_day_or_with_its_kt(env, tmp_path, monkeypatch):
    kept, other = tmp_path / "upload-1.mp4", tmp_path / "upload-2.mp4"
    for path, job_id in ((kept, "upload-job-0001"), (other, "upload-job-0002")):
        path.write_bytes(b"media")
        (tmp_path / f"{path.name}.wav").write_bytes(b"audio")
        env.JOB_QUEUE[job_id] = {"status": "queued", "progress": 0, "tenant_id": "local"}
        env.TASKS.enqueue(job_id, "upload", {"input_path": str(path), "media_format": "mp4"})
    _give_up(env, monkeypatch)
    assert env.TASKS.retryable("upload-job-0001") and env.TASKS.retryable("upload-job-0002")
    assert env.TASKS.purge_failed_inputs(now=time.time()) == 0               # within the day: kept
    assert env.TASKS.purge_failed_inputs(now=time.time() + 25 * 3600) == 2
    assert not kept.exists() and not (tmp_path / "upload-1.mp4.wav").exists()
    assert not env.TASKS.retryable("upload-job-0001") and env.TASKS.task("upload-job-0001")["payload"] == {}
    # Deleting a KT removes what it kept, without waiting for the day.
    other.write_bytes(b"media")
    env.JOB_QUEUE["upload-job-0003"] = {"status": "queued", "progress": 0, "tenant_id": "local"}
    env.TASKS.enqueue("upload-job-0003", "upload", {"input_path": str(other), "media_format": "mp4"})
    _give_up(env, monkeypatch)
    env.delete_job_data("upload-job-0003")
    assert not other.exists()


def test_a_restart_leaves_queued_kts_to_the_workers(env):
    queued, = _queue(env, 1)
    env.JOB_QUEUE["legacy-0001"] = {"status": "processing", "progress": 30}   # ran in-process before the queue
    assert env.JOB_STORE.mark_interrupted() == 1
    assert env.JOB_QUEUE[queued]["status"] == "queued"
    assert env.JOB_QUEUE["legacy-0001"]["error"] == INTERRUPTED_MESSAGE


def test_a_kt_deleted_while_it_runs_is_not_brought_back(client, env, monkeypatch):
    def deleted_midway(job_id, transcript, **kwargs):
        env.delete_job_data(job_id)                 # DELETE /jobs/{id} while the worker runs it
        _complete(env, job_id)                      # the pipeline still writes its result
    monkeypatch.setattr(env, "run_kt_pipeline", deleted_midway)
    job_id = _submit(client).json()["job_id"]
    Worker(worker_id="w1").drain(job_id)
    assert env.JOB_STORE.get(job_id) is None and env.JOB_STORE.current_document(job_id) is None


def test_a_worker_runs_at_most_its_concurrency(env, monkeypatch):
    active, peak, lock = [0], [0], threading.Lock()

    def slow_pipeline(job_id, transcript, **kwargs):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.3)
        with lock:
            active[0] -= 1
        _complete(env, job_id)

    monkeypatch.setattr(env, "run_kt_pipeline", slow_pipeline)
    monkeypatch.setattr(env, "build_job_pdf", lambda *a, **k: b"%PDF")
    ids = _queue(env, 5)
    worker = Worker(concurrency=2, worker_id="w-pool").start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline and any(env.JOB_STORE.get(i)["status"] != "completed" for i in ids):
            time.sleep(0.1)
    finally:
        worker.stop(timeout=10)
    assert all(env.JOB_STORE.get(i)["status"] == "completed" for i in ids)
    assert peak[0] == 2


def test_the_models_share_the_logical_cpus_among_the_kts_a_worker_runs(monkeypatch):
    import torch

    import worker

    calls = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)      # torch's default: the physical cores
    monkeypatch.setattr(worker.os, "cpu_count", lambda: 4)
    monkeypatch.setenv("CONTINUUM_TORCH_THREADS", "auto")
    assert worker.configure_torch_threads(1) == 4
    assert worker.configure_torch_threads(2) is None               # 2 each: never fewer than the default
    monkeypatch.setenv("CONTINUUM_TORCH_THREADS", "3")
    assert worker.configure_torch_threads(2) == 3
    monkeypatch.setenv("CONTINUUM_TORCH_THREADS", "0")
    assert worker.configure_torch_threads(1) is None
    assert calls == [4, 3]


def test_progress_written_by_a_worker_process_is_seen_by_the_api_process(env):
    api = PersistentJobs(env.JOB_STORE)
    api["shared-0001"] = {"status": "queued", "progress": 0}
    assert api["shared-0001"]["progress"] == 0                             # now cached in the API process
    worker_process = PersistentJobs(JobStore(env.JOB_STORE.path))
    worker_process.set_status("shared-0001", "processing")
    worker_process.set_progress("shared-0001", 60)
    assert api["shared-0001"]["status"] == "processing" and api["shared-0001"]["progress"] == 60


def test_an_api_only_process_is_ready_without_models(client, env, monkeypatch):
    monkeypatch.setenv("CONTINUUM_WORKERS", "0")
    monkeypatch.setattr(env, "MODEL", None)
    checks = client.get("/readyz").json()["checks"]
    assert "models_loaded" not in checks and checks["database_writable"] is True
    assert "ffmpeg" not in checks                     # the API runs no media tool (P1-4)
