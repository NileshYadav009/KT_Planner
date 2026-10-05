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
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DB_PATH = os.getenv(
    "CONTINUUM_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "continuum.sqlite"))
INTERRUPTED_MESSAGE = "The server restarted while this KT was being processed. Submit it again."
_ACTIVE = ("processing",)

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
"""


class JobStore:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(_SCHEMA)
            # Added with roles: who created the KT (an admin may delete any KT,
            # anyone else only their own).
            if "created_by" not in {row[1] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN created_by TEXT")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    @staticmethod
    def _pack(job: Dict[str, Any]) -> bytes:
        return zlib.compress(json.dumps(job, default=str).encode("utf-8"), 6)

    @staticmethod
    def _unpack(blob: Optional[bytes]) -> Dict[str, Any]:
        return json.loads(zlib.decompress(blob).decode("utf-8")) if blob else {}

    def put(self, job_id: str, job: Dict[str, Any]) -> None:
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
            return db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,)).rowcount > 0

    def delete_tenant(self, tenant_id: str) -> int:
        with self._lock, self._connect() as db:
            return db.execute("DELETE FROM jobs WHERE tenant_id = ?", (tenant_id,)).rowcount

    def mark_interrupted(self) -> int:
        """Jobs still 'processing' at startup were running in a process that
        no longer exists (they ran as in-process background tasks). Say so,
        rather than leaving them spinning forever."""
        interrupted = 0
        with self._connect() as db:
            rows = db.execute("SELECT job_id, payload FROM jobs WHERE status IN (%s)" % ",".join("?" * len(_ACTIVE)),
                              _ACTIVE).fetchall()
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
            db.execute("DELETE FROM jobs WHERE updated_at < ?", (cutoff,))
        return ids


class PersistentJobs(MutableMapping):
    """Dict-like job registry: an in-memory working set (progress updates of
    running jobs happen here) in front of the JobStore. Assigning a job
    writes it through; reading an unknown id loads it from the store."""

    def __init__(self, store: JobStore, cache_size: int = 64):
        self.store = store
        self._mem: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_size = cache_size

    def _remember(self, job_id: str, job: Dict[str, Any]) -> None:
        self._mem[job_id] = job
        self._mem.move_to_end(job_id)
        while len(self._mem) > self._cache_size:
            oldest, value = next(iter(self._mem.items()))
            if value.get("status") in _ACTIVE:
                break  # never evict a running job's live progress
            self._mem.popitem(last=False)

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        if job_id in self._mem:
            return self._mem[job_id]
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        self._remember(job_id, job)
        return job

    def __setitem__(self, job_id: str, job: Dict[str, Any]) -> None:
        previous = self._mem.get(job_id)
        if previous is not None and "tenant_id" not in job and previous.get("tenant_id"):
            job["tenant_id"] = previous["tenant_id"]
        self.store.put(job_id, job)
        self._remember(job_id, job)

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
        return job_id in self._mem or self.store.get(job_id) is not None
