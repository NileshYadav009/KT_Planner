"""P1-3 of the due-diligence review: one workspace cannot starve the others.
The queue takes turns between workspaces, and each workspace has daily
limits on KTs, recording minutes and LLM tokens; a submission over one is
refused with a message saying which limit and when the next one fits."""
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import quotas
from job_queue import TaskQueue
from job_store import JobStore, PersistentJobs

TRANSCRIPT = "The payments API runs on EKS and alerts go to PagerDuty."
DAY = 24 * 3600
NOW = 1_000_000.0


@pytest.fixture
def env(tmp_path, monkeypatch):
    """pipeline with its own job database, which the limits share."""
    import pipeline

    path = str(tmp_path / "continuum.sqlite")
    monkeypatch.setenv("CONTINUUM_DB_PATH", path)
    for var in quotas.ENV.values():
        monkeypatch.delenv(var, raising=False)
    store = JobStore(path)
    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setattr(pipeline, "TASKS", TaskQueue(store))
    return pipeline


def _queue(env, tenant, n):
    ids = []
    for i in range(n):
        job_id = f"{tenant}-{i:03d}"
        env.JOB_QUEUE[job_id] = {"status": "queued", "tenant_id": tenant}
        env.TASKS.enqueue(job_id, "transcript", {"transcript": TRANSCRIPT})
        ids.append(job_id)
    return ids


def _run_one(env, worker="w1"):
    task = env.TASKS.claim(worker)
    env.TASKS.finish(task["job_id"], worker)
    return task["job_id"]


# --------------------------------------------------------------------------
# Taking turns
# --------------------------------------------------------------------------

def test_ten_kts_of_one_workspace_hold_another_back_by_one_kt_at_most(env):
    # The review's acceptance test: before, globex's KT waited for all ten.
    acme = _queue(env, "acme", 10)
    globex = _queue(env, "globex", 1)[0]
    assert env.TASKS.position(globex) == 1
    order = [_run_one(env) for _ in range(11)]
    assert order.index(globex) == 1
    assert [j for j in order if j != globex] == acme          # each workspace's own KTs in the order they came
    assert env.TASKS.claim("w1") is None


def test_workspaces_alternate_while_both_have_kts_waiting(env):
    acme, globex = _queue(env, "acme", 3), _queue(env, "globex", 3)
    assert env.TASKS.position(globex[1]) == 3
    assert [_run_one(env) for _ in range(6)] == [acme[0], globex[0], acme[1], globex[1], acme[2], globex[2]]


def test_a_workspace_with_a_kt_running_does_not_take_the_next_free_worker(env):
    acme, globex = _queue(env, "acme", 3), _queue(env, "globex", 3)
    assert env.TASKS.claim("w1")["job_id"] == acme[0]
    assert env.TASKS.claim("w2")["job_id"] == globex[0]
    env.TASKS.finish(globex[0], "w2")
    # acme's first KT is still running: globex, with none running, goes next
    assert env.TASKS.claim("w2")["job_id"] == globex[1]


def test_one_workspace_runs_its_kts_in_the_order_they_came(env):
    ids = _queue(env, "local", 4)
    assert [env.TASKS.position(j) for j in ids] == [0, 1, 2, 3]
    assert [_run_one(env) for _ in range(4)] == ids
    assert env.TASKS.position(ids[0]) is None                  # no longer queued


# --------------------------------------------------------------------------
# Daily limits
# --------------------------------------------------------------------------

def test_a_kt_over_the_daily_limit_is_refused_with_when_the_next_one_fits(env, monkeypatch):
    monkeypatch.setenv("CONTINUUM_KTS_PER_DAY", "3")
    for i in range(3):
        quotas.admit("acme", f"j{i}", now=NOW + i * 600)
    with pytest.raises(quotas.LimitReached) as refused:
        quotas.admit("acme", "j3", now=NOW + 3600)
    assert refused.value.kind == "kts"
    # The first KT leaves the 24 hours at NOW + DAY.
    assert refused.value.retry_after == DAY - 3600 + 1
    assert refused.value.message.startswith(
        "This workspace has submitted 3 KTs and sessions in the last 24 hours, its daily limit of 3. "
        "The next one can be submitted in about 23 h (at ")
    assert refused.value.message.endswith("Your Continuum operator can raise the limit.")
    assert quotas.used("acme", NOW + 3600)["kts"] == 3          # the refusal is not counted
    assert quotas.admit("globex", "g1", now=NOW + 3600)           # nor does it touch another workspace
    assert quotas.admit("acme", "j4", now=NOW + DAY + 1)          # a day after the first, there is room


def test_recording_minutes_are_limited_per_day(env, monkeypatch):
    monkeypatch.setenv("CONTINUUM_RECORDING_MINUTES_PER_DAY", "90")
    quotas.admit("acme", "j1", minutes=60, now=NOW)
    with pytest.raises(quotas.LimitReached) as refused:
        quotas.admit("acme", "j2", minutes=45, now=NOW + 60)
    assert refused.value.kind == "minutes"
    assert ("This recording is 45 minutes long, and this workspace has used 60 of its 90 recording minutes "
            "in the last 24 hours. It can be submitted in about 23 h 59 min") in refused.value.message
    assert quotas.used("acme", NOW + 60) == {"kts": 1.0, "minutes": 60.0, "tokens": 0.0}
    quotas.admit("acme", "j3", minutes=30, now=NOW + 120)       # exactly fills the day
    with pytest.raises(quotas.LimitReached) as too_long:
        quotas.admit("globex", "g1", minutes=120, now=NOW)
    assert too_long.value.retry_after is None
    assert "This recording is 120 minutes long, more than this workspace's daily limit of 90" in too_long.value.message


def test_a_workspaces_own_limit_replaces_the_deployments(env, monkeypatch):
    monkeypatch.setenv("CONTINUUM_KTS_PER_DAY", "1")
    quotas.set_limits("acme", kts=0)                              # this workspace: no limit
    for i in range(3):
        quotas.admit("acme", f"a{i}", now=NOW)
    quotas.admit("globex", "g0", now=NOW)
    with pytest.raises(quotas.LimitReached):
        quotas.admit("globex", "g1", now=NOW)
    quotas.set_limits("acme", tokens=5000)
    assert quotas.own_limits("acme") == {"kts": 0, "minutes": None, "tokens": 5000}
    quotas.set_limits("acme", kts=None, tokens=None)              # back to the deployment's
    assert quotas.limits("acme") == {"kts": 1.0, "minutes": 0.0, "tokens": 0.0}
    with pytest.raises(ValueError):
        quotas.set_limits("acme", kts=-1)


def test_without_limits_every_submission_is_admitted_and_counted(env):
    for i in range(30):
        quotas.admit("local", f"j{i}", minutes=50, now=NOW)
    assert quotas.usage("local", NOW) == {"window_hours": 24, "kts": {"used": 30, "limit": None},
                                          "recording_minutes": {"used": 1500.0, "limit": None},
                                          "llm_tokens": {"used": 0, "limit": None}}


def test_a_submission_admitted_but_not_queued_is_taken_back(env, monkeypatch):
    monkeypatch.setenv("CONTINUUM_KTS_PER_DAY", "1")
    rows = quotas.admit("acme", "j1", minutes=10, now=NOW)
    quotas.release(rows)
    assert quotas.used("acme", NOW) == {"kts": 0.0, "minutes": 0.0, "tokens": 0.0}
    quotas.admit("acme", "j2", now=NOW)


def test_a_deleted_workspace_takes_its_limits_and_use_with_it(env):
    quotas.set_limits("acme", kts=5)
    quotas.admit("acme", "j1", now=NOW)
    quotas.record_tokens("acme", "j1", 1200, now=NOW)
    assert quotas.delete_tenant("acme") == 2
    assert quotas.own_limits("acme") == {"kts": None, "minutes": None, "tokens": None}


# --------------------------------------------------------------------------
# LLM tokens
# --------------------------------------------------------------------------

def test_a_kt_keeps_its_rules_only_results_once_its_workspace_used_the_days_tokens(env, tmp_path, monkeypatch):
    from llm.cached_provider import CachedLLMProvider, LLMBudgetExhausted, WORKSPACE_LIMIT_REASON
    from llm.usage import LLMUsageTracker, track_job

    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "llm_cache.sqlite"))
    monkeypatch.setenv("LLM_CACHE", "readwrite")
    monkeypatch.setenv("CONTINUUM_LLM_TOKENS_PER_DAY", "1000")
    quotas.record_tokens("acme", "earlier", 990)
    sent = []

    class Provider:
        model = "stub"

        def generate(self, prompt, **kwargs):
            sent.append(prompt)
            return "a reply of a few words"

    def _polish_one(provider, prompt):             # named as the real call site
        return provider.generate(prompt)

    provider = CachedLLMProvider(Provider())
    tracker = LLMUsageTracker("job")
    tracker.token_allowance = quotas.tokens_left("acme")
    assert tracker.token_allowance == 10
    with track_job(tracker):
        _polish_one(provider, "the first section's fragments, long enough to count more than ten tokens")
        with pytest.raises(LLMBudgetExhausted):
            _polish_one(provider, "the second section")
    assert len(sent) == 1
    stage = tracker.summary()["stages"][0]
    assert stage["skip_reasons"] == {WORKSPACE_LIMIT_REASON: 1}
    quotas.record_tokens("acme", "job", tracker.live_tokens())
    assert quotas.tokens_left("acme") < 0
    assert quotas.token_limit_notice("acme", 1) == (
        "This workspace reached its daily LLM limit (1,000 tokens in 24 hours): 1 LLM step(s) of this KT kept "
        "their rules-only result.")


def test_no_token_limit_leaves_the_llm_calls_alone(env):
    assert quotas.tokens_left("acme") is None


# --------------------------------------------------------------------------
# The API
# --------------------------------------------------------------------------

@pytest.fixture
def client(env, monkeypatch):
    from fastapi.testclient import TestClient
    import main

    monkeypatch.setattr(env, "run_kt_pipeline", lambda *args, **kwargs: None)
    return TestClient(main.app)          # no startup: no embedded worker


def _submit(client, key=None):
    headers = {"Idempotency-Key": key} if key else {}
    return client.post("/kt-from-transcript", json={"transcript": TRANSCRIPT, "force": True}, headers=headers)


def test_the_api_refuses_a_kt_over_the_limit_with_429_and_says_when(client, env, monkeypatch):
    monkeypatch.setenv("CONTINUUM_KTS_PER_DAY", "1")
    assert _submit(client).status_code == 200
    refused = _submit(client, key="second")
    assert refused.status_code == 429
    assert "its daily limit of 1. The next one can be submitted in about 24 h" in refused.json()["detail"]
    assert 0 < int(refused.headers["Retry-After"]) <= DAY + 1
    assert env.TASKS.stats()["queued"] == 1                               # nothing queued for it
    assert env.TASKS.reserve_key("local", "second", "next") == "next"     # its key is free for a later try
    assert client.get("/usage").json()["kts"] == {"used": 1, "limit": 1}


def test_a_recordings_minutes_are_counted_by_the_worker_and_a_refused_one_is_deleted(client, env, tmp_path,
                                                                                     monkeypatch):
    # The API runs no media tool (P1-4), so a recording's length is known
    # only when a worker probes it: over the limit, its KT fails with the
    # message, its upload is deleted and the KT it was counted as is taken back.
    import media_guard
    from worker import Worker

    monkeypatch.setenv("CONTINUUM_RECORDING_MINUTES_PER_DAY", "60")
    saved, ran = [], []

    async def save_upload(file):
        path = tmp_path / f"upload-{uuid.uuid4().hex}"
        path.write_bytes(b"audio")
        saved.append(path)
        return str(path)

    monkeypatch.setattr(media_guard, "save_upload", save_upload)
    monkeypatch.setattr(media_guard, "sniff", lambda path: "mp3")
    monkeypatch.setattr(media_guard, "probe", lambda path: {"format": "mp3", "duration": 45 * 60.0, "has_video": False})
    monkeypatch.setattr(env, "process_upload_task", lambda job_id, *args, **kwargs: ran.append(job_id))
    first = client.post("/upload", files={"file": ("kt.mp3", b"audio", "audio/mpeg")})
    second = client.post("/upload", files={"file": ("kt2.mp3", b"audio", "audio/mpeg")})
    assert first.status_code == second.status_code == 200                 # lengths not known yet
    first, second = first.json()["job_id"], second.json()["job_id"]
    Worker(worker_id="w1").drain(first, second)
    assert ran == [first]
    job = env.JOB_QUEUE[second]
    assert job["status"] == "failed"
    assert "This recording is 45 minutes long, and this workspace has used 45 of its 60" in job["error"]
    assert saved[0].exists() and not saved[1].exists()
    usage = client.get("/usage").json()
    assert usage["recording_minutes"] == {"used": 45.0, "limit": 60.0}
    assert usage["kts"]["used"] == 1                                      # the refused one taken back


def test_a_recordings_minutes_count_once_whatever_the_attempts(env, monkeypatch):
    # A worker that died after probing: the next attempt probes again.
    monkeypatch.setenv("CONTINUUM_RECORDING_MINUTES_PER_DAY", "60")
    quotas.admit("acme", "j1", minutes=45, count_kt=False, ref="upload-ref-1", now=NOW)
    assert quotas.admit("acme", "j1", minutes=45, count_kt=False, ref="upload-ref-1", now=NOW + 300) == []
    assert quotas.used("acme", NOW + 300)["minutes"] == 45


def test_the_cli_shows_and_sets_a_workspaces_limits(env, capsys):
    import auth

    tenant = auth.create_tenant("Acme")
    assert auth._main(["limits-set", tenant, "--kts", "20", "--tokens", "default"]) == 0
    assert quotas.own_limits(tenant) == {"kts": 20, "minutes": None, "tokens": None}
    out = capsys.readouterr().out
    assert "KTs and sessions" in out and "this workspace" in out
    assert auth._main(["limits-set", tenant, "--kts", "lots"]) == 2
    with pytest.raises(ValueError):
        auth._main(["limits", "t_unknown"])
