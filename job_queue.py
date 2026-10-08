"""The KT work queue (P1-4).

KT jobs used to run as FastAPI BackgroundTasks inside the API process: no
limit on how many ran at once, nothing queued survived a restart, and a
double submit paid for the work twice. Each submitted KT is now a row in the
tasks table of the job database. Workers (worker.py: threads in the API
process, or separate processes sharing the database) each claim one task at
a time under a lease:

    queued -> running -> done | failed

A worker renews the lease while the task runs. A task whose lease runs out
(the worker crashed, or was killed for running out of memory) is claimed
again, up to MAX_ATTEMPTS times. A failure inside the pipeline ("no speech
detected", "not a KT") is final: running it again gives the same answer.

SQLite suits one host (the API and its workers sharing a volume). The
interface (enqueue / claim / renew / finish) is what a Postgres or cloud
queue implementation would provide for more than one host.
"""
import json
import os
import re
import socket
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional

from job_store import JobStore

LEASE_SECONDS = float(os.getenv("CONTINUUM_TASK_LEASE_SECONDS", "120"))
MAX_ATTEMPTS = int(os.getenv("CONTINUUM_TASK_MAX_ATTEMPTS", "2"))
IDEMPOTENCY_TTL_SECONDS = 24 * 3600
# A worker that has not checked in for this long is gone.
WORKER_TIMEOUT_SECONDS = 60
GAVE_UP_MESSAGE = ("Processing stopped unexpectedly {attempts} times (the worker restarted or ran out of "
                   "memory). Submit the KT again; if it fails again, contact support.")

_KEY_RE = re.compile(r"[A-Za-z0-9_.:\-]{1,128}")


def valid_idempotency_key(key: Optional[str]) -> bool:
    return bool(key) and bool(_KEY_RE.fullmatch(key))


class TaskQueue:
    def __init__(self, store: JobStore):
        self.store = store        # same database; the tables are created by JobStore

    def _connect(self) -> sqlite3.Connection:
        # Autocommit, so each claim is one explicit BEGIN IMMEDIATE ... COMMIT.
        db = sqlite3.connect(self.store.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # -- submitting ---------------------------------------------------------

    def enqueue(self, job_id: str, kind: str, payload: Dict[str, Any]) -> None:
        """Queue a task for the job. A job whose last task finished can take
        another (a follow-up session, P2-1); one still queued or running
        cannot (ValueError)."""
        db = self._connect()
        try:
            changed = db.execute(
                """INSERT INTO tasks (job_id, kind, payload, state, enqueued_at) VALUES (?, ?, ?, 'queued', ?)
                   ON CONFLICT(job_id) DO UPDATE SET kind = excluded.kind, payload = excluded.payload,
                       state = 'queued', attempts = 0, enqueued_at = excluded.enqueued_at, started_at = NULL,
                       finished_at = NULL, lease_until = NULL, worker_id = NULL, error = NULL
                   WHERE tasks.state IN ('done', 'failed')""",
                (job_id, kind, json.dumps(payload), time.time())).rowcount
            if not changed:
                raise ValueError(f"job {job_id} already has a task queued or running")
        finally:
            db.close()

    def reserve_key(self, tenant_id: str, key: str, job_id: str) -> str:
        """Bind an Idempotency-Key to a new job. Returns the job the key is
        bound to: `job_id` if this request won, the earlier job otherwise."""
        now = time.time()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM idempotency_keys WHERE created_at < ?", (now - IDEMPOTENCY_TTL_SECONDS,))
            db.execute("INSERT OR IGNORE INTO idempotency_keys (tenant_id, key, job_id, created_at) VALUES (?, ?, ?, ?)",
                       (tenant_id, key, job_id, now))
            row = db.execute("SELECT job_id FROM idempotency_keys WHERE tenant_id = ? AND key = ?",
                             (tenant_id, key)).fetchone()
            db.execute("COMMIT")
            return row[0]
        except Exception:
            db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def release_key(self, tenant_id: str, key: str, job_id: str) -> None:
        """Free a key whose request was refused, so a corrected retry can use it."""
        db = self._connect()
        try:
            db.execute("DELETE FROM idempotency_keys WHERE tenant_id = ? AND key = ? AND job_id = ?",
                       (tenant_id, key, job_id))
        finally:
            db.close()

    # -- working ------------------------------------------------------------

    def claim(self, worker_id: str, job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """The oldest queued task (or one whose worker's lease ran out), now
        leased to `worker_id`; only that job's task when `job_id` is given.
        None when there is nothing to do."""
        now = time.time()
        gave_up: List[tuple] = []
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            # A task that has already been tried MAX_ATTEMPTS times and lost
            # its worker again fails instead of being tried forever.
            gave_up = db.execute("SELECT job_id, attempts FROM tasks WHERE state = 'running' AND lease_until < ? "
                                 "AND attempts >= ?", (now, MAX_ATTEMPTS)).fetchall()
            for job_id, attempts in gave_up:
                db.execute("UPDATE tasks SET state = 'failed', finished_at = ?, payload = '{}', lease_until = NULL, "
                           "error = ? WHERE job_id = ?", (now, GAVE_UP_MESSAGE.format(attempts=attempts), job_id))
            row = db.execute("SELECT job_id, kind, payload, attempts, enqueued_at FROM tasks WHERE (state = 'queued' "
                             "OR (state = 'running' AND lease_until < ?)) AND (? IS NULL OR job_id = ?) "
                             "ORDER BY enqueued_at LIMIT 1", (now, job_id, job_id)).fetchone()
            if row:
                db.execute("UPDATE tasks SET state = 'running', worker_id = ?, attempts = attempts + 1, "
                           "started_at = ?, lease_until = ? WHERE job_id = ?",
                           (worker_id, now, now + LEASE_SECONDS, row[0]))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        finally:
            db.close()
        for job_id, attempts in gave_up:
            self._fail_job(job_id, GAVE_UP_MESSAGE.format(attempts=attempts))
        if not row:
            return None
        return {"job_id": row[0], "kind": row[1], "payload": json.loads(row[2]), "attempt": row[3] + 1,
                "enqueued_at": row[4]}

    def _fail_job(self, job_id: str, message: str) -> None:
        job = self.store.get(job_id)
        if job is not None:
            job.update(status="failed", error=message)
            self.store.put(job_id, job)

    def renew(self, worker_id: str, job_ids: List[str]) -> None:
        if not job_ids:
            return
        db = self._connect()
        try:
            db.execute("UPDATE tasks SET lease_until = ? WHERE worker_id = ? AND state = 'running' AND job_id IN (%s)"
                       % ",".join("?" * len(job_ids)), [time.time() + LEASE_SECONDS, worker_id, *job_ids])
        finally:
            db.close()

    def finish(self, job_id: str, worker_id: str, error: Optional[str] = None) -> bool:
        """Mark the task done (or failed). The payload, which can hold the
        transcript, is cleared. False when the task no longer exists: the KT
        was deleted while it ran."""
        db = self._connect()
        try:
            return db.execute("UPDATE tasks SET state = ?, finished_at = ?, payload = '{}', lease_until = NULL, "
                              "error = ? WHERE job_id = ? AND worker_id = ?",
                              ("failed" if error else "done", time.time(), error, job_id, worker_id)).rowcount > 0
        finally:
            db.close()

    def heartbeat(self, worker_id: str, concurrency: int, running: int) -> None:
        now = time.time()
        db = self._connect()
        try:
            db.execute("""INSERT INTO workers (worker_id, concurrency, running, started_at, last_seen)
                          VALUES (?, ?, ?, ?, ?)
                          ON CONFLICT(worker_id) DO UPDATE SET concurrency = excluded.concurrency,
                              running = excluded.running, last_seen = excluded.last_seen""",
                       (worker_id, concurrency, running, now, now))
            db.execute("DELETE FROM workers WHERE last_seen < ?", (now - 10 * WORKER_TIMEOUT_SECONDS,))
        finally:
            db.close()

    def remove_worker(self, worker_id: str) -> None:
        db = self._connect()
        try:
            db.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
        finally:
            db.close()

    # -- reading ------------------------------------------------------------

    def task(self, job_id: str) -> Optional[Dict[str, Any]]:
        db = self._connect()
        try:
            row = db.execute("SELECT kind, payload, state, attempts, enqueued_at, error FROM tasks WHERE job_id = ?",
                             (job_id,)).fetchone()
        finally:
            db.close()
        if not row:
            return None
        return {"kind": row[0], "payload": json.loads(row[1]), "state": row[2], "attempts": row[3],
                "enqueued_at": row[4], "error": row[5]}

    def position(self, job_id: str) -> Optional[int]:
        """How many queued KTs are ahead of this one (None unless it is queued)."""
        db = self._connect()
        try:
            row = db.execute("SELECT enqueued_at FROM tasks WHERE job_id = ? AND state = 'queued'", (job_id,)).fetchone()
            if not row:
                return None
            return db.execute("SELECT COUNT(*) FROM tasks WHERE state = 'queued' AND enqueued_at < ?",
                              (row[0],)).fetchone()[0]
        finally:
            db.close()

    def stats(self) -> Dict[str, Any]:
        """Queue depth, the age of the oldest waiting KT and the live
        workers: for /readyz and the metrics endpoint."""
        now = time.time()
        db = self._connect()
        try:
            queued, oldest = db.execute("SELECT COUNT(*), MIN(enqueued_at) FROM tasks WHERE state = 'queued'").fetchone()
            running = db.execute("SELECT COUNT(*) FROM tasks WHERE state = 'running'").fetchone()[0]
            workers, slots = db.execute("SELECT COUNT(*), COALESCE(SUM(concurrency), 0) FROM workers "
                                        "WHERE last_seen >= ?", (now - WORKER_TIMEOUT_SECONDS,)).fetchone()
        finally:
            db.close()
        return {"queued": queued, "running": running, "oldest_queued_seconds": round(now - oldest, 1) if oldest else 0.0,
                "workers": workers, "worker_slots": slots}


def worker_alive_on_this_host(db_path: str) -> bool:
    """A worker started on this host (container) checked in recently."""
    with sqlite3.connect(db_path, timeout=5) as db:
        row = db.execute("SELECT MAX(last_seen) FROM workers WHERE worker_id LIKE ?",
                         (socket.gethostname() + "-%",)).fetchone()
    return bool(row and row[0] and row[0] >= time.time() - WORKER_TIMEOUT_SECONDS)


if __name__ == "__main__":
    # Container health check for a worker: `python job_queue.py --worker-alive`.
    # Imports no models, so it answers in well under a second.
    from job_store import DEFAULT_DB_PATH

    if sys.argv[1:] != ["--worker-alive"]:
        raise SystemExit("usage: python job_queue.py --worker-alive")
    raise SystemExit(0 if worker_alive_on_this_host(DEFAULT_DB_PATH) else 1)
