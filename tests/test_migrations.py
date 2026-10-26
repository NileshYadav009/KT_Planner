"""Numbered database migrations (enterprise checklist P1; migrations.py).
Before, columns were added by ad-hoc checks in two modules, with no record
of what a database had been through."""
import logging
import os
import sqlite3
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import migrations  # noqa: E402
from job_store import JobStore  # noqa: E402

# The tables as a release before these columns created them.
_OLD_LAYOUT = """
CREATE TABLE jobs (job_id TEXT PRIMARY KEY, tenant_id TEXT, status TEXT NOT NULL, title TEXT,
                   created_at REAL NOT NULL, updated_at REAL NOT NULL, error TEXT, payload BLOB);
CREATE TABLE documents (job_id TEXT PRIMARY KEY, pdf BLOB NOT NULL, rendered_at REAL NOT NULL);
CREATE TABLE sessions (session_hash TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, key_label TEXT,
                       role TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL);
INSERT INTO jobs VALUES ('old-1', 'acme', 'completed', 'Old KT', 1, 2, NULL, NULL);
INSERT INTO documents VALUES ('old-1', x'25504446', 3);
INSERT INTO sessions VALUES ('h', 'acme', 'alice', 'admin', 1, 9999999999);
"""


def _columns(path, table):
    with sqlite3.connect(path) as db:
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def test_the_migrations_are_numbered_in_order():
    assert [m[0] for m in migrations.MIGRATIONS] == list(range(1, migrations.LATEST + 1))


def test_an_older_database_is_brought_up_to_date_and_keeps_its_data(tmp_path):
    path = str(tmp_path / "old.sqlite")
    with sqlite3.connect(path) as db:
        db.executescript(_OLD_LAYOUT)
    store = JobStore(path)                                   # migrates on open
    assert "created_by" in _columns(path, "jobs")
    assert {"user_id", "last_seen_at"} <= _columns(path, "sessions")
    assert "job_version" in _columns(path, "documents")      # the old PDF cache rebuilt
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT title FROM jobs WHERE job_id = 'old-1'").fetchone() == ("Old KT",)
        assert db.execute("SELECT key_label FROM sessions").fetchone() == ("alice",)
    assert [v for v, _, _ in migrations.applied(path)] == list(range(1, migrations.LATEST + 1))
    assert store.current_document("old-1") is None
    assert migrations.migrate(path) == []                    # nothing twice


def test_a_new_database_records_every_migration(tmp_path, monkeypatch):
    import auth

    path = str(tmp_path / "new.sqlite")
    monkeypatch.setenv("CONTINUUM_DB_PATH", path)
    JobStore(path)
    auth.create_tenant("Fresh")                              # auth's tables, created as they are now
    assert {"user_id", "last_seen_at"} <= _columns(path, "sessions")
    assert len(migrations.applied(path)) == migrations.LATEST


def test_processes_migrating_at_once_apply_each_migration_once(tmp_path):
    path = str(tmp_path / "race.sqlite")
    with sqlite3.connect(path) as db:
        db.executescript(_OLD_LAYOUT)
    results = []
    threads = [threading.Thread(target=lambda: results.append(migrations.migrate(path))) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(v for ran in results for v in ran) == list(range(1, migrations.LATEST + 1))


def test_a_database_from_a_newer_release_is_reported(tmp_path, caplog):
    path = str(tmp_path / "newer.sqlite")
    migrations.migrate(path)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO schema_migrations VALUES (?, 'from a newer release', 0)", (migrations.LATEST + 1,))
    with caplog.at_level(logging.ERROR, logger="migrations"):
        assert migrations.migrate(path) == []
    assert f"migration {migrations.LATEST + 1}, newer than this release" in caplog.text


def test_the_status_command_lists_each_migration(tmp_path, monkeypatch, capsys):
    path = str(tmp_path / "status.sqlite")
    monkeypatch.setenv("CONTINUUM_DB_PATH", path)
    with sqlite3.connect(path) as db:
        db.executescript(_OLD_LAYOUT)
    migrations.main([])
    assert capsys.readouterr().out.count("PENDING") == migrations.LATEST
    migrations.main(["--apply"])
    out = capsys.readouterr().out
    assert "PENDING" not in out and "sessions.last_seen_at" in out
