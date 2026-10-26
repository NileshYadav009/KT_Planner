#!/usr/bin/env python3
"""Back up and restore Continuum's data: the job database and screenshots.

    python scripts/backup.py backup --dest /backups [--keep 14]
    python scripts/backup.py verify /backups/continuum-20261026T020000Z
    python scripts/backup.py restore /backups/continuum-20261026T020000Z [--force]
    python scripts/backup.py drill [--dest DIR]     # back up, restore into a scratch folder, read it all back

The database (CONTINUUM_DB_PATH: KTs, document versions, the queue, people,
API keys, sessions, the audit log) is copied with SQLite's online backup, so
the API and workers keep running while it is taken. The screenshots
(KT_ASSETS_DIR) are archived beside it, and a manifest records each file's
SHA-256 and the row count of every table. The LLM response cache is not
included: it refills itself. A backup is a folder; copy it off the host (an
object store or another machine) as the next step of the same job.

restore checks the backup first (checksums, SQLite integrity check), then
puts it in place. Stop the API and the workers before restoring. Whatever it
replaces is kept beside it (".before-restore-<time>"), not deleted.

docs/OPERATIONS.md has the schedule and the RPO and RTO this gives.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PREFIX = "continuum-"
DB_FILE = "continuum.sqlite"
ASSETS_FILE = "kt_assets.tar.gz"
MANIFEST = "manifest.json"


def db_path() -> str:
    from job_store import DEFAULT_DB_PATH
    return os.getenv("CONTINUUM_DB_PATH", DEFAULT_DB_PATH)


def assets_dir() -> str:
    # pipeline.KT_ASSETS_DIR, without importing the pipeline and its models.
    return os.getenv("KT_ASSETS_DIR", os.path.join(ROOT, "data", "kt_assets"))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_only(path: str) -> str:
    return pathlib.Path(path).resolve().as_uri() + "?mode=ro"


def _table_counts(path: str) -> Dict[str, int]:
    db = sqlite3.connect(_read_only(path), uri=True)
    try:
        names = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE 'kt_search_%' ORDER BY name")]
        return {name: db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in names}
    finally:
        db.close()


def _integrity(path: str) -> str:
    db = sqlite3.connect(_read_only(path), uri=True)
    try:
        return db.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        db.close()


def backup(dest: str, keep: int = 14, source_db: Optional[str] = None, source_assets: Optional[str] = None) -> str:
    """A new backup folder in `dest`; the oldest beyond `keep` are removed."""
    source_db = source_db or db_path()
    source_assets = source_assets or assets_dir()
    if not os.path.exists(source_db):
        raise SystemExit(f"No database at {source_db}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final = os.path.join(dest, PREFIX + stamp)
    partial = final + ".partial"
    os.makedirs(partial, exist_ok=False)
    started = time.time()

    target = os.path.join(partial, DB_FILE)
    src = sqlite3.connect(_read_only(source_db), uri=True, timeout=30)    # never writes to it
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)                       # a consistent copy while others write
    finally:
        dst.close()
        src.close()
    check = _integrity(target)
    if check != "ok":
        raise SystemExit(f"The copy failed SQLite's integrity check: {check}")
    manifest = {
        "created_at": stamp, "source_database": source_db, "source_assets": source_assets,
        "database": {"file": DB_FILE, "sha256": _sha256(target), "bytes": os.path.getsize(target),
                     "tables": _table_counts(target)},
        "assets": None,
    }
    if os.path.isdir(source_assets):
        archive = os.path.join(partial, ASSETS_FILE)
        files = 0
        with tarfile.open(archive, "w:gz") as tar:
            for folder, _, names in os.walk(source_assets):
                for name in sorted(names):
                    full = os.path.join(folder, name)
                    tar.add(full, arcname=os.path.relpath(full, source_assets))
                    files += 1
        manifest["assets"] = {"file": ASSETS_FILE, "sha256": _sha256(archive), "files": files}
    manifest["seconds"] = round(time.time() - started, 1)
    with open(os.path.join(partial, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(partial, final)                # only a complete backup has its final name

    backups = sorted(n for n in os.listdir(dest) if n.startswith(PREFIX) and not n.endswith(".partial"))
    for old in backups[:max(0, len(backups) - keep)]:
        shutil.rmtree(os.path.join(dest, old))
    return final


def verify(folder: str) -> dict:
    """The backup's manifest, after its checksums and the database's
    integrity check pass; SystemExit naming the problem otherwise."""
    try:
        with open(os.path.join(folder, MANIFEST), encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{folder} is not a complete backup: {exc}")
    parts = [manifest["database"]] + ([manifest["assets"]] if manifest.get("assets") else [])
    for part in parts:
        path = os.path.join(folder, part["file"])
        if not os.path.exists(path) or _sha256(path) != part["sha256"]:
            raise SystemExit(f"{part['file']} is missing or does not match its checksum")
    check = _integrity(os.path.join(folder, DB_FILE))
    if check != "ok":
        raise SystemExit(f"The database failed SQLite's integrity check: {check}")
    return manifest


def _set_aside(path: str, stamp: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    kept = f"{path}.before-restore-{stamp}"
    os.replace(path, kept)
    return kept


def restore(folder: str, target_db: Optional[str] = None, target_assets: Optional[str] = None,
            force: bool = False) -> dict:
    """Put a verified backup in place; what it replaces is kept beside it."""
    manifest = verify(folder)
    target_db = target_db or db_path()
    target_assets = target_assets or assets_dir()
    if os.path.exists(target_db) and not force:
        raise SystemExit(f"{target_db} exists. Stop the API and the workers, then run again with --force "
                         "(the current database is kept beside it).")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs(os.path.dirname(os.path.abspath(target_db)), exist_ok=True)
    kept = [_set_aside(target_db, stamp)] + [_set_aside(target_db + s, stamp) for s in ("-wal", "-shm")]
    shutil.copyfile(os.path.join(folder, DB_FILE), target_db)
    if manifest.get("assets"):
        kept.append(_set_aside(target_assets, stamp))
        os.makedirs(target_assets, exist_ok=True)
        with tarfile.open(os.path.join(folder, ASSETS_FILE), "r:gz") as tar:
            tar.extractall(target_assets, filter="data")
    if _table_counts(target_db) != manifest["database"]["tables"]:
        raise SystemExit("The restored database does not hold what the backup recorded")
    return {"manifest": manifest, "kept": [k for k in kept if k]}


def drill(dest: Optional[str] = None) -> dict:
    """Back up the live data, restore it into a scratch folder and read every
    KT back through the job store, as the app would."""
    from job_store import JobStore

    started = time.time()
    scratch = tempfile.mkdtemp(prefix="continuum-drill-")
    try:
        folder = backup(dest or os.path.join(scratch, "backups"), keep=10 ** 6)
        restored_db = os.path.join(scratch, "restored", DB_FILE)
        restored_assets = os.path.join(scratch, "restored", "kt_assets")
        result = restore(folder, restored_db, restored_assets)
        store = JobStore(restored_db)
        db = sqlite3.connect(restored_db)
        ids = [r[0] for r in db.execute("SELECT job_id FROM jobs")]
        db.close()
        unreadable = [job_id for job_id in ids if not isinstance(store.get(job_id), dict)]
        if unreadable:
            raise SystemExit(f"{len(unreadable)} restored KT(s) could not be read, first {unreadable[0]}")
        assets = result["manifest"].get("assets") or {}
        restored_files = sum(len(names) for _, _, names in os.walk(restored_assets))
        if assets and restored_files != assets["files"]:
            raise SystemExit(f"{restored_files} screenshot file(s) restored of {assets['files']}")
        return {"backup": folder, "jobs": len(ids), "tables": result["manifest"]["database"]["tables"],
                "asset_files": restored_files, "seconds": round(time.time() - started, 1)}
    finally:
        if dest is None:
            shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backup", help="take a backup")
    b.add_argument("--dest", required=True, help="folder that holds the backups")
    b.add_argument("--keep", type=int, default=14, help="backups to keep (default 14)")
    v = sub.add_parser("verify", help="check a backup")
    v.add_argument("folder")
    r = sub.add_parser("restore", help="restore a backup (stop the API and workers first)")
    r.add_argument("folder")
    r.add_argument("--force", action="store_true", help="replace the current data (kept beside it)")
    d = sub.add_parser("drill", help="back up, restore into a scratch folder, read every KT back")
    d.add_argument("--dest", help="keep the drill's backup here (default: removed afterwards)")
    args = parser.parse_args(argv)

    if args.command == "backup":
        folder = backup(args.dest, args.keep)
        manifest = verify(folder)
        print(f"Backup {folder}: {manifest['database']['tables'].get('jobs', 0)} KTs, "
              f"{(manifest.get('assets') or {}).get('files', 0)} screenshot file(s), {manifest['seconds']} s")
    elif args.command == "verify":
        manifest = verify(args.folder)
        print(f"{args.folder} is complete: {manifest['database']['tables'].get('jobs', 0)} KTs, taken {manifest['created_at']}")
    elif args.command == "restore":
        result = restore(args.folder, force=args.force)
        print(f"Restored {args.folder} to {db_path()} and {assets_dir()}.")
        for kept in result["kept"]:
            print(f"  kept the replaced data at {kept}")
    else:
        result = drill(args.dest)
        print(f"Restore drill passed: {result['jobs']} KTs read back, {result['asset_files']} screenshot file(s), "
              f"{len(result['tables'])} tables, {result['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
