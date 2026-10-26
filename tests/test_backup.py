"""Backups with a tested restore (enterprise checklist P0; scripts/backup.py).
Before, nothing backed up the one database that holds every KT, person and
key. All on scratch data."""
import json
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backup  # noqa: E402
from job_store import JobStore  # noqa: E402


@pytest.fixture
def data(tmp_path, monkeypatch):
    db = str(tmp_path / "data" / "continuum.sqlite")
    assets = tmp_path / "data" / "kt_assets"
    (assets / "job-1").mkdir(parents=True)
    (assets / "job-1" / "screen_01.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    store = JobStore(db)
    store.put("job-1", {"status": "completed", "tenant_id": "acme", "transcript": "The API runs on EKS."})
    store.put("job-2", {"status": "completed", "tenant_id": "globex", "transcript": "Backups run nightly."})
    monkeypatch.setenv("CONTINUUM_DB_PATH", db)
    monkeypatch.setenv("KT_ASSETS_DIR", str(assets))
    return {"db": db, "assets": str(assets), "store": store, "dest": str(tmp_path / "backups")}


def test_a_backup_restores_to_the_same_data(data, tmp_path):
    folder = backup.backup(data["dest"])
    manifest = backup.verify(folder)
    assert manifest["database"]["tables"]["jobs"] == 2 and manifest["assets"]["files"] == 1

    target = str(tmp_path / "elsewhere" / "continuum.sqlite")
    assets = str(tmp_path / "elsewhere" / "kt_assets")
    backup.restore(folder, target, assets)
    restored = JobStore(target)
    assert restored.get("job-2")["transcript"] == "Backups run nightly."
    assert open(os.path.join(assets, "job-1", "screen_01.jpg"), "rb").read().startswith(b"\xff\xd8")


def test_restore_keeps_what_it_replaces_and_needs_force(data):
    folder = backup.backup(data["dest"])
    data["store"].put("job-3", {"status": "completed", "tenant_id": "acme", "transcript": "Added after."})
    with pytest.raises(SystemExit, match="--force"):
        backup.restore(folder)
    result = backup.restore(folder, force=True)
    assert JobStore(data["db"]).get("job-3") is None                   # back to the backup
    kept = [k for k in result["kept"] if os.path.basename(k).startswith("continuum.sqlite.before-restore")]
    assert kept and JobStore(kept[0]).get("job-3")["transcript"] == "Added after."


def test_a_damaged_or_partial_backup_is_refused(data):
    folder = backup.backup(data["dest"])
    with open(os.path.join(folder, "continuum.sqlite"), "r+b") as fh:
        fh.seek(200)
        fh.write(b"\x00\x01\x02")
    with pytest.raises(SystemExit, match="checksum"):
        backup.verify(folder)
    os.remove(os.path.join(folder, "manifest.json"))
    with pytest.raises(SystemExit, match="not a complete backup"):
        backup.restore(folder, force=True)
    assert JobStore(data["db"]).get("job-1")                           # nothing was replaced


def test_old_backups_are_pruned(data, monkeypatch):
    stamps = iter(["20261001T020000Z", "20261002T020000Z", "20261003T020000Z"])
    monkeypatch.setattr(backup.time, "strftime", lambda fmt, t=None: next(stamps))
    for _ in range(3):
        backup.backup(data["dest"], keep=2)
    assert sorted(os.listdir(data["dest"])) == ["continuum-20261002T020000Z", "continuum-20261003T020000Z"]


def test_the_backup_never_writes_to_the_live_database(data):
    before = os.path.getmtime(data["db"])
    backup.backup(data["dest"])
    assert os.path.getmtime(data["db"]) == before


def test_the_restore_drill_reads_every_kt_back(data, capsys):
    assert backup.main(["drill"]) == 0
    out = capsys.readouterr().out
    assert "Restore drill passed: 2 KTs read back, 1 screenshot file(s)" in out
    assert not os.path.exists(data["dest"])                             # the drill cleans up after itself
