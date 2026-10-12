"""P3-3: a finished KT nobody has changed for CONTINUUM_KT_REVIEW_DAYS days is
due for review: the list says so, and the team channel gets one reminder per
quiet period, naming the incoming owner."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import staleness
from job_store import JobStore

DAY = 86400.0
NOW = 1_800_000_000.0


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTINUUM_REVIEW_WEBHOOK_URL", "https://hooks.example/review")
    monkeypatch.delenv("CONTINUUM_KT_REVIEW_DAYS", raising=False)
    return JobStore(str(tmp_path / "jobs.sqlite"))


def _put(store, job_id, days_ago, status="completed", **job):
    store.put(job_id, dict({"status": status, "tenant_id": "t1", "title": f"{job_id} KT"}, **job))
    with store._connect() as db:
        db.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (NOW - days_ago * DAY, job_id))


def test_a_finished_kt_left_alone_past_the_interval_is_due():
    assert staleness.annotate({"status": "completed", "updated_at": NOW - 200 * DAY}, NOW)["review_due"]
    assert not staleness.annotate({"status": "completed", "updated_at": NOW - 10 * DAY}, NOW)["review_due"]
    assert not staleness.annotate({"status": "processing", "updated_at": NOW - 400 * DAY}, NOW)["review_due"]
    due = staleness.annotate({"status": "completed_with_warnings", "updated_at": NOW - 200 * DAY}, NOW)
    assert due["review_due_at"] == pytest.approx(NOW - 20 * DAY)


def test_the_interval_is_configurable_and_zero_turns_it_off(monkeypatch):
    monkeypatch.setenv("CONTINUUM_KT_REVIEW_DAYS", "30")
    assert staleness.annotate({"status": "completed", "updated_at": NOW - 31 * DAY}, NOW)["review_due"]
    monkeypatch.setenv("CONTINUUM_KT_REVIEW_DAYS", "0")
    assert not staleness.annotate({"status": "completed", "updated_at": NOW - 999 * DAY}, NOW)["review_due"]


def test_one_reminder_per_quiet_period_naming_the_incoming_owner(store):
    _put(store, "old", 200, signoff={"participants": {"receiver": {"email": "rita@example.com"}}})
    _put(store, "fresh", 5)
    _put(store, "running", 300, status="processing")
    sent = []
    notify = lambda text, **fields: sent.append((text, fields)) or True  # noqa: E731

    assert staleness.send_reminders(store, store.get, NOW, notify) == 1
    text, fields = sent[0]
    assert fields["job_id"] == "old" and fields["owner"] == "rita@example.com"
    assert text.startswith("old KT has not been reviewed for 200 days") and "rita@example.com, please" in text

    # Not again for the same quiet period, by this worker or another.
    assert staleness.send_reminders(store, store.get, NOW + DAY, notify) == 0
    # A change starts a new period; once that is over, one more reminder.
    _put(store, "old", -1, signoff={"participants": {}}, created_by="gina@example.com")
    assert staleness.send_reminders(store, store.get, NOW + 2 * DAY, notify) == 0
    assert staleness.send_reminders(store, store.get, NOW + 200 * DAY, notify) == 2     # "fresh" is due too now
    assert {f["job_id"]: f["owner"] for _, f in sent[-2:]}["old"] == "gina@example.com"


def test_a_failed_post_is_tried_again_and_no_webhook_sends_nothing(store, monkeypatch):
    _put(store, "old", 200)
    assert staleness.send_reminders(store, store.get, NOW, lambda *a, **k: False) == 0
    assert staleness.send_reminders(store, store.get, NOW, lambda *a, **k: True) == 1

    _put(store, "other", 300)
    monkeypatch.delenv("CONTINUUM_REVIEW_WEBHOOK_URL")
    assert staleness.send_reminders(store, store.get, NOW, lambda *a, **k: True) == 0
    assert [r["job_id"] for r in store.due_for_review(NOW - 180 * DAY)] == ["other"]   # nothing claimed


def test_reminders_are_checked_at_most_hourly(monkeypatch):
    calls = []
    monkeypatch.setattr(staleness, "send_reminders", lambda *a, **k: calls.append(1) or 0)
    monkeypatch.setattr(staleness, "_last_check", 0.0)
    now = time.time()
    staleness.check_due_reviews(now)
    staleness.check_due_reviews(now + 60)
    staleness.check_due_reviews(now + 3601)
    assert len(calls) == 2


def test_the_kt_list_says_which_are_due(store, monkeypatch):
    from fastapi.testclient import TestClient
    import main
    import pipeline
    from job_store import PersistentJobs

    monkeypatch.setattr(pipeline, "JOB_STORE", store)
    monkeypatch.setattr(pipeline, "JOB_QUEUE", PersistentJobs(store))
    monkeypatch.setenv("CONTINUUM_AUTH", "disabled")
    store.put("a", {"status": "completed", "tenant_id": "local", "title": "A"})
    store.put("b", {"status": "completed", "tenant_id": "local", "title": "B"})
    with store._connect() as db:
        db.execute("UPDATE jobs SET updated_at = ? WHERE job_id = 'a'", (time.time() - 400 * DAY,))
    jobs = {j["job_id"]: j for j in TestClient(main.app).get("/jobs").json()["jobs"]}
    assert jobs["a"]["review_due"] and not jobs["b"]["review_due"]
