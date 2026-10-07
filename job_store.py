"""Persistent job and document storage (P0-2).

Jobs used to live only in a Python dict: a restart or deploy erased every KT,
and the uploaded recording was already deleted, so nothing could be rebuilt.
Every job is now written to SQLite (results compressed; the transcript and
segments are kept, so a document can always be re-rendered). JOB_QUEUE in
pipeline.py stays a dict-like object so the rest of the code is unchanged:
reads fall through to the database, writes go to both.

SQLite suits a single-instance deployment or pilot. The interface (get / put /
list / delete) is what a Postgres implementation would provide later.
"""
import json
import os
import sqlite3
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, List, Optional, Tuple

DEFAULT_DB_PATH = os.getenv(
    "CONTINUUM_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "continuum.sqlite"))
INTERRUPTED_MESSAGE = "The server restarted while this KT was being processed. Submit it again."
_ACTIVE = ("queued", "processing")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    tenant_id   TEXT,
    status      TEXT NOT NULL,
    title       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    error       TEXT,
    payload     BLOB
);
CREATE INDEX IF NOT EXISTS jobs_tenant_created ON jobs(tenant_id, created_at DESC);

-- P1-4: the work queue (job_queue.py). One row per submitted KT; the payload
-- (the transcript, or the path of the uploaded file) is cleared when it ends.
CREATE TABLE IF NOT EXISTS tasks (
    job_id       TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL,
    state        TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    enqueued_at  REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    lease_until  REAL,
    worker_id    TEXT,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS tasks_state_enqueued ON tasks(state, enqueued_at);
CREATE TABLE IF NOT EXISTS idempotency_keys (
    tenant_id   TEXT NOT NULL,
    key         TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tenant_id, key)
);
CREATE TABLE IF NOT EXISTS workers (
    worker_id    TEXT PRIMARY KEY,
    concurrency  INTEGER NOT NULL,
    running      INTEGER NOT NULL,
    started_at   REAL NOT NULL,
    last_seen    REAL NOT NULL
);
-- P1-6: one row per finished KT, for metrics (observability.py). Numbers
-- and ids only; kept when the KT is deleted, removed with its workspace.
CREATE TABLE IF NOT EXISTS job_runs (
    job_id       TEXT PRIMARY KEY,
    tenant_id    TEXT,
    status       TEXT NOT NULL,
    finished_at  REAL NOT NULL,
    record       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS job_runs_finished ON job_runs(finished_at);
-- P1-1: every saved state of the document (version 1 is what the pipeline
-- generated; each reviewer save adds one). Compressed JSON snapshot.
CREATE TABLE IF NOT EXISTS document_versions (
    job_id      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    author      TEXT,
    summary     TEXT,
    snapshot    BLOB NOT NULL,
    PRIMARY KEY (job_id, version)
);
-- The finished PDF, rendered by the worker, so a download is a read.
-- job_version is the job's updated_at it was rendered from.
CREATE TABLE IF NOT EXISTS documents (
    job_id       TEXT PRIMARY KEY,
    pdf          BLOB NOT NULL,
    job_version  REAL NOT NULL
);
"""
# Rows that belong to a job and go when it goes.
_JOB_TABLES = ("tasks", "idempotency_keys", "documents", "document_versions")


class JobStore:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            # The stored PDFs were first keyed by render time (rendered_at),
            # and CREATE TABLE IF NOT EXISTS never changes an existing table,
            # so a database from then failed every download ("no such column:
            # job_version"). The rows are only a cache of rendered PDFs: drop
            # the old table and the next download renders again.
            columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            if columns and "job_version" not in columns:
                db.execute("DROP TABLE documents")
            db.executescript(_SCHEMA)
            # Added with roles: who created the KT (an admin may delete any KT,
            # anyone else only their own).
            if "created_by" not in {row[1] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN created_by TEXT")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def writable(self) -> bool:
        """True when a write transaction can be opened (readiness check)."""
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("ROLLBACK")
            return True
        except sqlite3.Error:
            return False

    @staticmethod
    def _pack(job: Dict[str, Any]) -> bytes:
        return zlib.compress(json.dumps(job, default=str).encode("utf-8"), 6)

    @staticmethod
    def _unpack(blob: Optional[bytes]) -> Dict[str, Any]:
        return json.loads(zlib.decompress(blob).decode("utf-8")) if blob else {}

    def put(self, job_id: str, job: Dict[str, Any]) -> float:
        """Insert or replace a job. Returns its new version (updated_at)."""
        now = time.time()
        ko = job.get("knowledge_object") or {}
        title = ko.get("system_name") or job.get("title")
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO jobs (job_id, tenant_id, status, title, created_at, updated_at, error, payload, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,
                       title=COALESCE(excluded.title, jobs.title), updated_at=excluded.updated_at,
                       error=excluded.error, payload=excluded.payload,
                       tenant_id=COALESCE(jobs.tenant_id, excluded.tenant_id),
                       created_by=COALESCE(jobs.created_by, excluded.created_by)""",
                # A job created outside the API (scripts, tests) belongs to the
                # "local" tenant, the same one auth.ensure_job_access assumes.
                (job_id, job.get("tenant_id") or "local", job.get("status", "processing"), title, now, now,
                 job.get("error"), self._pack(job), job.get("created_by_id")),
            )
        return now

    def updated_at(self, job_id: str) -> Optional[float]:
        """The job's version: changes on every write, from any process."""
        with self._connect() as db:
            row = db.execute("SELECT updated_at FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row[0] if row else None

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT tenant_id, payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        job = self._unpack(row[1])
        job.setdefault("tenant_id", row[0])
        return job

    def created_at(self, job_id: str) -> Optional[float]:
        with self._connect() as db:
            row = db.execute("SELECT created_at FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row[0] if row else None

    def tenant_of(self, job_id: str) -> Optional[str]:
        with self._connect() as db:
            row = db.execute("SELECT tenant_id FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row[0] if row else None

    def list(self, tenant_id: Optional[str], limit: int = 50) -> List[Dict[str, Any]]:
        sql = "SELECT job_id, status, title, created_at, updated_at, error, created_by FROM jobs"
        args: list = []
        if tenant_id is not None:
            sql += " WHERE tenant_id = ?"
            args.append(tenant_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(int(limit))
        with self._connect() as db:
            rows = db.execute(sql, args).fetchall()
        return [{"job_id": r[0], "status": r[1], "title": r[2], "created_at": r[3], "updated_at": r[4],
                 "error": r[5], "created_by_id": r[6]} for r in rows]

    def delete(self, job_id: str) -> bool:
        with self._lock, self._connect() as db:
            for table in _JOB_TABLES:
                db.execute(f"DELETE FROM {table} WHERE job_id = ?", (job_id,))
            return db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,)).rowcount > 0

    def delete_tenant(self, tenant_id: str) -> int:
        with self._lock, self._connect() as db:
            for table in _JOB_TABLES:
                db.execute(f"DELETE FROM {table} WHERE job_id IN (SELECT job_id FROM jobs WHERE tenant_id = ?)",
                           (tenant_id,))
            db.execute("DELETE FROM idempotency_keys WHERE tenant_id = ?", (tenant_id,))
            db.execute("DELETE FROM job_runs WHERE tenant_id = ?", (tenant_id,))
            return db.execute("DELETE FROM jobs WHERE tenant_id = ?", (tenant_id,)).rowcount

    def record_run(self, run: Dict[str, Any]) -> None:
        """Keep one finished KT's numbers (observability.run_record)."""
        with self._lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO job_runs (job_id, tenant_id, status, finished_at, record) "
                       "VALUES (?, ?, ?, ?, ?)",
                       (run["job_id"], run.get("tenant_id"), run["status"], run["finished_at"], json.dumps(run)))

    def runs(self, since: float = 0.0) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT record FROM job_runs WHERE finished_at >= ? ORDER BY finished_at",
                              (since,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def save_version(self, job_id: str, version: int, author: str, summary: str, snapshot: Dict[str, Any]) -> bool:
        """Store one document version. False if that version already exists
        (another save got there first)."""
        with self._lock, self._connect() as db:
            return db.execute("INSERT OR IGNORE INTO document_versions (job_id, version, created_at, author, summary, "
                              "snapshot) VALUES (?, ?, ?, ?, ?, ?)",
                              (job_id, int(version), time.time(), author, summary, self._pack(snapshot))).rowcount > 0

    def versions(self, job_id: str) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT version, created_at, author, summary FROM document_versions WHERE job_id = ? "
                              "ORDER BY version", (job_id,)).fetchall()
        return [{"version": r[0], "created_at": r[1], "author": r[2], "summary": r[3]} for r in rows]

    def version(self, job_id: str, version: int) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT snapshot FROM document_versions WHERE job_id = ? AND version = ?",
                             (job_id, int(version))).fetchone()
        return self._unpack(row[0]) if row else None

    def save_document(self, job_id: str, pdf: bytes, job_version: float) -> None:
        """Store a rendered PDF with the job version (updated_at) it shows."""
        with self._lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO documents (job_id, pdf, job_version) VALUES (?, ?, ?)",
                       (job_id, sqlite3.Binary(pdf), job_version))

    def current_document(self, job_id: str) -> Optional[bytes]:
        """The stored PDF, if the job has not changed since it was rendered
        (a correction makes it stale, and the next download re-renders)."""
        with self._connect() as db:
            row = db.execute("""SELECT d.pdf FROM documents d JOIN jobs j ON j.job_id = d.job_id
                                WHERE d.job_id = ? AND d.job_version = j.updated_at""", (job_id,)).fetchone()
        return bytes(row[0]) if row else None

    def mark_interrupted(self) -> int:
        """Jobs still active at startup with no task in the queue ran as
        in-process background tasks before the queue existed, in a process
        that is gone. Say so, rather than leaving them spinning forever. Jobs
        with a queued or running task are left alone: a worker resumes them."""
        interrupted = 0
        with self._connect() as db:
            rows = db.execute("SELECT job_id, payload FROM jobs WHERE status IN (%s) AND job_id NOT IN "
                              "(SELECT job_id FROM tasks WHERE state IN ('queued', 'running'))"
                              % ",".join("?" * len(_ACTIVE)), _ACTIVE).fetchall()
        for job_id, blob in rows:
            job = self._unpack(blob)
            job.update(status="failed", error=INTERRUPTED_MESSAGE)
            self.put(job_id, job)
            interrupted += 1
        return interrupted

    def purge_older_than(self, days: float) -> List[str]:
        cutoff = time.time() - days * 86400
        with self._lock, self._connect() as db:
            ids = [r[0] for r in db.execute("SELECT job_id FROM jobs WHERE updated_at < ?", (cutoff,)).fetchall()]
            for table in _JOB_TABLES:
                db.execute(f"DELETE FROM {table} WHERE job_id IN (SELECT job_id FROM jobs WHERE updated_at < ?)",
                           (cutoff,))
            db.execute("DELETE FROM jobs WHERE updated_at < ?", (cutoff,))
        return ids


class PersistentJobs(MutableMapping):
    """Dict-like job registry: an in-memory working set in front of the
    JobStore. Assigning a job writes it through. A cached copy is returned
    only while the database still holds that version, so a job written by
    another process (a worker, P1-4) is never served stale."""

    def __init__(self, store: JobStore, cache_size: int = 64):
        self.store = store
        # job_id -> (version, job)
        self._mem: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
        self._cache_size = cache_size

    def _remember(self, job_id: str, job: Dict[str, Any], version: float) -> None:
        self._mem[job_id] = (version, job)
        self._mem.move_to_end(job_id)
        while len(self._mem) > self._cache_size:
            oldest, (_, value) = next(iter(self._mem.items()))
            if value.get("status") in _ACTIVE:
                break  # keep a running job's working copy
            self._mem.popitem(last=False)

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        version = self.store.updated_at(job_id)
        if version is None:
            self._mem.pop(job_id, None)
            raise KeyError(job_id)
        cached = self._mem.get(job_id)
        if cached is not None and cached[0] == version:
            self._mem.move_to_end(job_id)
            return cached[1]
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        # Read after the version: a write in between only makes the next read reload.
        self._remember(job_id, job, version)
        return job

    def __setitem__(self, job_id: str, job: Dict[str, Any]) -> None:
        cached = self._mem.get(job_id)
        previous = cached[1] if cached else None
        if previous is not None and "tenant_id" not in job and previous.get("tenant_id"):
            job["tenant_id"] = previous["tenant_id"]
        self._remember(job_id, job, self.store.put(job_id, job))

    def set_progress(self, job_id: str, progress: int) -> None:
        """Record a running job's progress where every process can read it."""
        self._set_field(job_id, "progress", progress)

    def set_status(self, job_id: str, status: str) -> None:
        self._set_field(job_id, "status", status)

    def _set_field(self, job_id: str, name: str, value: Any) -> None:
        try:
            job = self[job_id]
        except KeyError:
            return
        job[name] = value
        self[job_id] = job

    def __delitem__(self, job_id: str) -> None:
        self._mem.pop(job_id, None)
        if not self.store.delete(job_id):
            raise KeyError(job_id)

    def pop(self, job_id, *default):
        try:
            value = self[job_id]
        except KeyError:
            if default:
                return default[0]
            raise
        del self[job_id]
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._mem))

    def __len__(self) -> int:
        return len(self._mem)

    def __contains__(self, job_id) -> bool:
        return self.store.updated_at(job_id) is not None
