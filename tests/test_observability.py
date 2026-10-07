"""P1-6: structured logs with job and tenant ids, per-stage metrics, an
alert when a KT fails, and error tracking that keeps transcripts out."""
import http.server
import json
import logging
import os
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import observability
from job_queue import TaskQueue
from job_store import JobStore, PersistentJobs
from worker import Worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPT = "Never restart the ledger writer during the batch window."


@pytest.fixture
def env(tmp_path, monkeypatch):
    import pipeline

    store = JobStore(str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setattr(pipeline, "TASKS", TaskQueue(store))
    monkeypatch.setattr(pipeline, "build_job_pdf", lambda *a, **k: b"%PDF")
    return pipeline


def _ko(missing=(), mentioned=()):
    from kt_schema_loader import SCHEMA

    titles = {s["id"]: s["title"] for s in SCHEMA}
    rows = [{"Domain": titles[s], "Coverage": "Missing"} for s in missing]
    return {"system_name": "Payments", "rendered_sections": [], "_mentioned_elsewhere": {s: [] for s in mentioned},
            "sections": [{"id": "kt_coverage", "_coverage_rows": rows}]}


def _run_one(env, monkeypatch, job_id="obs-job-0001", outcome=None):
    """Queue one KT and run it on a worker with a stand-in pipeline that
    passes two stage checkpoints and ends in `outcome`."""
    outcome = outcome or {"status": "completed", "knowledge_object": _ko(["security_controls", "cost_optimization"],
                                                                         ["security_controls"]),
                          "llm_usage": {"llm_calls": 12, "failed": 1, "rejected": 2, "input_tokens": 900,
                                        "output_tokens": 300},
                          "stage_errors": [{"stage": "Text polishing", "error": "Timeout"}]}

    def fake_pipeline(job_id, transcript, **kwargs):
        logging.getLogger("pipeline").warning("classifying")
        observability.checkpoint("Classification")
        observability.checkpoint("Rendering")
        env.JOB_QUEUE[job_id] = dict(outcome)

    monkeypatch.setattr(env, "run_kt_pipeline", fake_pipeline)
    env.JOB_QUEUE[job_id] = {"status": "queued", "tenant_id": "acme"}
    env.TASKS.enqueue(job_id, "transcript", {"transcript": TRANSCRIPT})
    Worker(worker_id="obs").drain(job_id)
    return job_id


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------

def test_log_lines_carry_the_job_and_tenant_they_were_written_for():
    formatter = observability.JsonFormatter()
    record = logging.LogRecord("pipeline", logging.WARNING, __file__, 1, "Polishing failed: %s", ("Timeout",), None)
    with observability.job_context("job-1234", "acme"):
        observability._ContextFilter().filter(record)
    line = json.loads(formatter.format(record))
    assert line["message"] == "Polishing failed: Timeout"
    assert line["job_id"] == "job-1234" and line["tenant_id"] == "acme" and line["level"] == "WARNING"

    outside = logging.LogRecord("api", logging.INFO, __file__, 1, "hello", (), None)
    observability._ContextFilter().filter(outside)
    assert "job_id" not in json.loads(formatter.format(outside))


def test_each_request_gets_an_id_in_its_logs_and_response():
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    assert len(client.get("/healthz").headers["X-Request-ID"]) == 16
    assert client.get("/healthz", headers={"X-Request-ID": "lb-7f3a"}).headers["X-Request-ID"] == "lb-7f3a"
    assert client.get("/healthz", headers={"X-Request-ID": "bad id\n"}).headers["X-Request-ID"] != "bad id\n"


# --------------------------------------------------------------------------
# Run records and metrics
# --------------------------------------------------------------------------

def test_a_finished_kt_records_its_stage_times_and_counts(env, monkeypatch):
    job_id = _run_one(env, monkeypatch)
    run, = env.JOB_STORE.runs()
    assert run["job_id"] == job_id and run["status"] == "completed" and run["tenant_id"] == "acme"
    assert {"Classification", "Rendering", "Saving", "PDF rendering"} <= set(run["stage_seconds"])
    assert run["failed_stages"] == ["Text polishing"]
    assert (run["llm_calls"], run["llm_failed"], run["llm_rejected"]) == (12, 1, 2)
    assert (run["sections_missing"], run["missing_but_mentioned"]) == (2, 1)
    assert "ledger writer" not in json.dumps(run)                         # no transcript content


def test_metrics_are_served_only_to_the_metrics_token(env, monkeypatch):
    from fastapi.testclient import TestClient
    import main

    _run_one(env, monkeypatch)
    client = TestClient(main.app)
    monkeypatch.delenv("CONTINUUM_METRICS_TOKEN", raising=False)
    assert client.get("/metrics").status_code == 404
    monkeypatch.setenv("CONTINUUM_METRICS_TOKEN", "scrape-secret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    body = client.get("/metrics", headers={"Authorization": "Bearer scrape-secret"}).text
    assert 'continuum_jobs_total{status="completed"} 1' in body
    assert 'continuum_stage_runs_total{stage="Classification"} 1' in body
    assert 'continuum_stage_failures_total{stage="Text polishing"} 1' in body
    assert 'continuum_llm_fallbacks_total{reason="rejected"} 2' in body
    assert "continuum_sections_missing_but_mentioned_total 1" in body
    assert 'continuum_queue_depth{state="queued"} 0' in body
    assert 'continuum_job_duration_seconds_bucket{le="+Inf"} 1' in body


def test_the_ops_report_shows_stage_latency_and_failures(env, monkeypatch):
    _run_one(env, monkeypatch)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ops_report.py")], capture_output=True,
                         text=True, env=dict(os.environ, CONTINUUM_DB_PATH=env.JOB_STORE.path), timeout=120)
    assert out.returncode == 0, out.stderr
    assert "KTs: 1 — 1 completed" in out.stdout
    assert "Text polishing" in out.stdout and "Classification" in out.stdout


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------

@pytest.fixture
def webhook(monkeypatch):
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("CONTINUUM_ALERT_WEBHOOK_URL", f"http://127.0.0.1:{server.server_port}/hook")
    yield received
    server.shutdown()


def test_an_alert_fires_when_a_kt_fails(env, monkeypatch, webhook):
    _run_one(env, monkeypatch, "obs-fail-0001", {"status": "failed", "error": "Document rendering failed: KeyError"})
    alert, = webhook
    assert alert["alert"] == "kt_failed" and alert["job_id"] == "obs-fail-0001" and alert["tenant_id"] == "acme"
    assert "Document rendering failed" in alert["text"]


def test_no_alert_when_the_input_was_not_a_kt(env, monkeypatch, webhook):
    _run_one(env, monkeypatch, "obs-input-0001",
             {"status": "failed", "error": "No speech detected in the uploaded file.", "failure": "input"})
    assert webhook == []
    run, = env.JOB_STORE.runs()
    assert run["failure"] == "input"


def test_an_empty_upload_is_an_input_failure(env, tmp_path, monkeypatch):
    src = tmp_path / "empty.upload"
    src.write_bytes(b"")
    env.JOB_QUEUE["obs-empty-0001"] = {"status": "processing"}
    env.process_upload_task("obs-empty-0001", str(src), str(src) + ".mp3")
    job = env.JOB_QUEUE["obs-empty-0001"]
    assert job["status"] == "failed" and job["failure"] == "input"


def test_a_queue_backlog_alerts_once_an_hour(monkeypatch, webhook):
    monkeypatch.setattr(observability, "_queue_alert_sent", 0.0)
    monkeypatch.setenv("CONTINUUM_ALERT_QUEUE_AGE_SECONDS", "600")
    assert observability.check_queue_age({"oldest_queued_seconds": 300, "queued": 2, "workers": 1}, now=10_000) is False
    assert observability.check_queue_age({"oldest_queued_seconds": 1200, "queued": 9, "workers": 0}, now=10_000) is True
    assert observability.check_queue_age({"oldest_queued_seconds": 1300, "queued": 9, "workers": 0}, now=10_060) is False
    assert len(webhook) == 1 and webhook[0]["alert"] == "queue_age" and "20 min" in webhook[0]["text"]


# --------------------------------------------------------------------------
# Error tracking
# --------------------------------------------------------------------------

def test_sentry_gets_no_request_bodies_local_variables_or_personal_data(monkeypatch):
    import sentry_sdk

    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert observability.init_error_tracking() is False and calls == []

    monkeypatch.setenv("SENTRY_DSN", "https://public@sentry.example/1")
    assert observability.init_error_tracking() is True
    options, = calls
    assert options["send_default_pii"] is False and options["include_local_variables"] is False
    assert options["max_request_body_size"] == "never"
    with observability.job_context("job-77", "acme"):
        event = options["before_send"]({"request": {"data": "TRANSCRIPT TEXT"}, "message": "boom"}, {})
    assert "request" not in event and event["tags"] == {"job_id": "job-77", "tenant_id": "acme"}
