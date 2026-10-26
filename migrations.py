"""Numbered migrations of the job database (CONTINUUM_DB_PATH).

    python migrations.py            # which migrations this database has, and which are pending
    python migrations.py --apply    # apply the pending ones (the app also does, when it starts)

Each module creates its own tables (CREATE TABLE IF NOT EXISTS), written as
they are now. A change to a table that may already exist (a column, a
rebuild, an index, a data fix) is a migration here: numbered, applied once,
in order, and recorded in schema_migrations with the time it ran. Only add
to the end of MIGRATIONS, and never edit one that has shipped. A migration
skips a table that does not exist yet: it will be created as it is now.

Migrations go forward only. To roll back a release, restore the backup
taken before upgrading (scripts/backup.py; docs/OPERATIONS.md). An older
release started on a database a newer one migrated logs an error naming
the migration it does not know.
"""
import argparse
import logging
import os
import sqlite3
import sys
import time
from typing import Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _columns(db: sqlite3.Connection, table: str) -> Set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _add_column(table: str, column: str, declaration: str) -> Callable[[sqlite3.Connection], None]:
    def apply(db: sqlite3.Connection) -> None:
        columns = _columns(db, table)
        if columns and column not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    return apply


def _documents_by_job_version(db: sqlite3.Connection) -> None:
    # The stored PDFs were first keyed by render time (rendered_at), so a
    # database from then failed every download ("no such column:
    # job_version"). The rows are only a cache: the next download renders.
    columns = _columns(db, "documents")
    if columns and "job_version" not in columns:
        db.execute("DROP TABLE documents")
        db.execute("CREATE TABLE documents (job_id TEXT PRIMARY KEY, pdf BLOB NOT NULL, job_version REAL NOT NULL)")


MIGRATIONS: Tuple[Tuple[int, str, Callable[[sqlite3.Connection], None]], ...] = (
    (1, "documents: stored PDFs keyed by the job version they show", _documents_by_job_version),
    (2, "jobs.created_by: who created the KT (roles)", _add_column("jobs", "created_by", "TEXT")),
    (3, "sessions.user_id: the person a session belongs to", _add_column("sessions", "user_id", "TEXT")),
    (4, "sessions.last_seen_at: idle sign-out", _add_column("sessions", "last_seen_at", "REAL")),
)
LATEST = MIGRATIONS[-1][0]

_SCHEMA = """CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_at  REAL NOT NULL
)"""


def _connect(path: str) -> sqlite3.Connection:
    from job_store import ClosingConnection

    # Autocommit: each migration is one explicit BEGIN IMMEDIATE ... COMMIT.
    db = sqlite3.connect(path, timeout=30, isolation_level=None, factory=ClosingConnection)
    db.execute(_SCHEMA)
    return db


def applied(path: str) -> List[Tuple[int, str, float]]:
    with _connect(path) as db:
        return db.execute("SELECT version, name, applied_at FROM schema_migrations ORDER BY version").fetchall()


def migrate(path: str) -> List[int]:
    """Apply the pending migrations, in order; the versions applied now.
    Safe for several processes at once: each migration is applied by the
    first to take the write lock."""
    done = {version for version, _, _ in applied(path)}
    unknown = sorted(v for v in done if v > LATEST)
    if unknown:
        logger.error("The database at %s has migration %d, newer than this release knows (%d): run the newer "
                     "release, or restore the backup taken before the upgrade", path, unknown[-1], LATEST)
    ran: List[int] = []
    with _connect(path) as db:
        for version, name, apply in MIGRATIONS:
            if version in done:
                continue
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute("SELECT 1 FROM schema_migrations WHERE version = ?", (version,)).fetchone():
                    db.execute("ROLLBACK")              # another process applied it meanwhile
                    continue
                apply(db)
                db.execute("INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                           (version, name, time.time()))
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
            ran.append(version)
            logger.info("Database migration %d applied: %s", version, name)
    return ran


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="apply the pending migrations")
    args = parser.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from job_store import DEFAULT_DB_PATH

    path = os.getenv("CONTINUUM_DB_PATH", DEFAULT_DB_PATH)
    if args.apply:
        print(f"Applied: {migrate(path) or 'nothing, the database is up to date'}")
    done = {v: (name, at) for v, name, at in applied(path)}
    print(f"Database: {path}")
    for version, name, _ in MIGRATIONS:
        state = time.strftime("applied %Y-%m-%d %H:%M", time.localtime(done[version][1])) if version in done else "PENDING"
        print(f"  {version:>3}  {state:<24}  {name}")
    for version in sorted(v for v in done if v > LATEST):
        print(f"  {version:>3}  {'unknown to this release':<24}  {done[version][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
