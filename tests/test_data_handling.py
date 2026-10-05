"""P0-9: customer data handling. Before: spoken or pasted credentials went to
the LLM, the cache and the PDF; the LLM cache was shared by every customer;
customer sentences were copied into one global vocabulary file; nothing could
be deleted per job or per customer."""
import json
import os
import sqlite3
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from redaction import redact_secrets


@pytest.mark.parametrize("text,secret", [
    ("The admin password is Sup3rS3cret! so be careful.", "Sup3rS3cret!"),
    ("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCY", "wJalrXUtnFEMI"),
    ("key id AKIAIOSFODNN7EXAMPLE was used", "AKIAIOSFODNN7EXAMPLE"),
    ("connect with postgres://app:hunter2pw@db.internal:5432/orders", "hunter2pw"),
    ("Use the key gsk_abcdefghijklmnop1234 for tests", "gsk_abcdefghijklmnop1234"),
    ("token: 'abc123xyz'", "abc123xyz"),
])
def test_credentials_are_redacted(text, secret):
    redacted, n = redact_secrets(text)
    assert secret not in redacted and n >= 1


@pytest.mark.parametrize("text", [
    "The token is rotated every 90 days.",
    "The password is stored in Key Vault.",
    "Escalate to Priya Nair at priya@acme.io or +44 20 7946 0958.",
    "The TOKEN_TTL is configured centrally.",
])
def test_operational_text_is_left_alone(text):
    assert redact_secrets(text) == (text, 0)


TRANSCRIPT = (
    "This KT is for TripWise, the booking backend behind our travel app. "
    "Bookings are written to DynamoDB and payments go through Stripe. "
    "Datadog is our monitoring tool and PagerDuty pages us if booking success drops below ninety five percent. "
    "RTO is four hours. The database admin password is Sup3rS3cret! and the deploy key is AKIAIOSFODNN7EXAMPLE. "
    "Never delete items from the bookings table by hand. "
    "Escalate to Marco Silva if PagerDuty is not acknowledged within fifteen minutes."
)


class Recorder:
    model = "recorder"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt + str(kwargs.get("system_prompt") or ""))
        return "NOT_MENTIONED"


@pytest.fixture(scope="module")
def mapper():
    from context_mapper import ContextMappingPipeline
    from kt_schema_loader import SCHEMA
    return ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)


def _run(monkeypatch, mapper, provider, job_id, tenant_id="local", cache_path=None):
    import ai
    import pipeline
    from devops_transcription import clean_transcript
    from llm.cached_provider import CachedLLMProvider

    monkeypatch.setenv("LLM_CACHE", "readwrite" if cache_path else "off")
    if cache_path:
        monkeypatch.setenv("LLM_CACHE_PATH", cache_path)
    wrapped = CachedLLMProvider(provider)
    monkeypatch.setattr(pipeline, "get_llm_provider", lambda: wrapped if _allowed() else None)
    monkeypatch.setattr(ai, "get_llm_provider", lambda: wrapped if _allowed() else None)
    monkeypatch.setattr(pipeline, "record_candidates", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "MAPPER_PIPELINE", mapper)
    pipeline.JOB_QUEUE[job_id] = {"status": "processing", "progress": 0, "tenant_id": tenant_id}
    return pipeline.run_kt_pipeline(job_id, clean_transcript(TRANSCRIPT))


def _allowed():
    from llm.tenant_context import llm_allowed
    return llm_allowed()


def test_secrets_never_reach_the_llm_the_store_or_the_document(monkeypatch, mapper):
    import pipeline

    rec = Recorder()
    result = _run(monkeypatch, mapper, rec, "data-secret-0001")
    assert rec.prompts, "the fake LLM should have been called"
    everything = "\n".join(rec.prompts) + json.dumps(pipeline.JOB_STORE.get("data-secret-0001"))
    assert "Sup3rS3cret" not in everything and "AKIAIOSFODNN7EXAMPLE" not in everything
    assert any("redacted" in n for n in result["notices"])
    pipeline.JOB_STORE.delete("data-secret-0001")


def test_llm_cache_entries_are_per_tenant_and_purgeable(monkeypatch, mapper, tmp_path):
    import pipeline
    from llm import cache

    cache_path = str(tmp_path / "cache.sqlite")
    first, second = Recorder(), Recorder()
    _run(monkeypatch, mapper, first, "data-cache-a-0001", tenant_id="tenant-a", cache_path=cache_path)
    _run(monkeypatch, mapper, second, "data-cache-b-0001", tenant_id="tenant-b", cache_path=cache_path)
    # Identical transcript, different tenant: nothing is reused across tenants.
    assert len(second.prompts) == len(first.prompts) > 0
    rows = dict(sqlite3.connect(cache_path).execute("SELECT tenant, COUNT(*) FROM llm_responses GROUP BY tenant").fetchall())
    assert set(rows) == {"tenant-a", "tenant-b"}
    monkeypatch.setenv("LLM_CACHE_PATH", cache_path)
    assert cache.purge_tenant("tenant-a") == rows["tenant-a"]
    for job_id in ("data-cache-a-0001", "data-cache-b-0001"):
        pipeline.JOB_STORE.delete(job_id)


def test_a_tenant_can_forbid_external_llm_calls(monkeypatch, mapper):
    import auth
    import pipeline

    tenant = auth.create_tenant("No-LLM Corp", llm_policy="none")
    rec = Recorder()
    result = _run(monkeypatch, mapper, rec, "data-nollm-0001", tenant_id=tenant)
    assert rec.prompts == []
    assert result["status"] == "completed"
    assert any("External LLM disabled" in n for n in result["notices"])
    pipeline.JOB_STORE.delete("data-nollm-0001")


def test_vocabulary_file_holds_terms_only_and_survives_concurrent_writes(tmp_path):
    from vocabulary_learning import record_candidates

    path = str(tmp_path / "candidates.json")
    cands = [{"phrase": f"Term{i}", "signal": "near_miss", "example_sentence": f"Secret customer sentence {i}."}
             for i in range(20)]
    threads = [threading.Thread(target=record_candidates, args=([c], f"job-{i}"), kwargs={"path": path})
               for i, c in enumerate(cands)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    data = json.load(open(path, encoding="utf-8"))
    assert len(data) == 20
    text = json.dumps(data)
    assert "Secret customer sentence" not in text and "job-" not in text


def test_deleting_a_job_removes_its_record_and_screenshots(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import main
    import pipeline

    monkeypatch.setattr(pipeline, "KT_ASSETS_DIR", str(tmp_path))
    pipeline.JOB_QUEUE["data-del-0001"] = {"status": "completed"}
    os.makedirs(tmp_path / "data-del-0001")
    (tmp_path / "data-del-0001" / "screen_01.jpg").write_bytes(b"x")
    resp = TestClient(main.app).delete("/jobs/data-del-0001")
    assert resp.status_code == 200
    assert pipeline.JOB_STORE.get("data-del-0001") is None
    assert not (tmp_path / "data-del-0001").exists()


def test_deleting_a_tenant_removes_everything_it_owns(monkeypatch, tmp_path):
    import auth
    import pipeline
    from llm import cache
    from llm.tenant_context import tenant_scope

    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("LLM_CACHE", "readwrite")
    tenant = auth.create_tenant("Leaving Ltd")
    key = auth.create_key(tenant, "k")
    auth.add_user(tenant, "leaver@leaving.example", "giver")
    pipeline.JOB_QUEUE["data-tenant-0001"] = {"status": "completed", "tenant_id": tenant}
    with tenant_scope(tenant):
        cache.put(cache.make_key(provider="p", model="m", prompt="x", params={}), response="r", call_site="s",
                  provider="p", model="m", input_tokens=1, output_tokens=1)
    report = auth.delete_tenant(tenant)
    assert report == {"tenant_id": tenant, "jobs": 1, "cached_llm_responses": 1, "api_keys": 1, "people": 1}
    assert auth.list_users(tenant) == []
    assert pipeline.JOB_STORE.get("data-tenant-0001") is None
    assert auth.principal_for_key(key) is None


def test_retention_deletes_jobs_older_than_the_limit(monkeypatch):
    import pipeline

    pipeline.JOB_QUEUE["data-old-0001"] = {"status": "completed"}
    with sqlite3.connect(pipeline.JOB_STORE.path) as db:
        db.execute("UPDATE jobs SET updated_at = ? WHERE job_id = 'data-old-0001'", (time.time() - 40 * 86400,))
    monkeypatch.setenv("CONTINUUM_RETENTION_DAYS", "30")
    assert pipeline.apply_retention() >= 1
    assert pipeline.JOB_STORE.get("data-old-0001") is None
