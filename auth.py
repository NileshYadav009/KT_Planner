"""Tenants, API keys and request authentication (P0-1).

Before this module every route was public: any caller could read, change or
export any job, and /status returned the full transcript to anyone with a
job id.

Model: a tenant (a customer) has API keys; every job belongs to the tenant
whose key created it, and is invisible to every other tenant (404, so
existence is not revealed). Keys are shown once and stored only as SHA-256
hashes. SSO (OIDC) for people can be layered on later by issuing the same
Principal from a verified ID token.

    python -m auth create-tenant "Acme Transition Team"   # prints tenant id + first key
    python -m auth create-key <tenant_id> [label]
    python -m auth list
    python -m auth revoke <key prefix>
    python -m auth llm-policy <tenant_id> none|default   # "none": no external LLM
    python -m auth delete-tenant <tenant_id>            # jobs, screenshots, cache, keys

CONTINUUM_AUTH=disabled turns authentication off for local development only
(every request then acts as tenant "local"); it is logged loudly.
"""
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request

from job_store import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

KEY_PREFIX = "ckt_"
ASSET_URL_TTL_SECONDS = int(os.getenv("CONTINUUM_ASSET_URL_TTL", "3600"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    llm_policy  TEXT NOT NULL DEFAULT 'default'
);
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash    TEXT PRIMARY KEY,
    key_prefix  TEXT NOT NULL,
    tenant_id   TEXT NOT NULL REFERENCES tenants(tenant_id),
    label       TEXT,
    role        TEXT NOT NULL DEFAULT 'member',
    created_at  REAL NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    session_hash TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    key_label    TEXT,
    role         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    name  TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_warned_disabled = False
_lock = threading.Lock()


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    role: str
    key_label: str
    llm_policy: str = "default"


def _db_path() -> str:
    return os.getenv("CONTINUUM_DB_PATH", DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(_db_path())), exist_ok=True)
    db = sqlite3.connect(_db_path(), timeout=30)
    db.executescript(_SCHEMA)
    return db


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def auth_enabled() -> bool:
    return os.getenv("CONTINUUM_AUTH", "required").strip().lower() not in ("disabled", "off", "0", "false")


# --------------------------------------------------------------------------
# Administration
# --------------------------------------------------------------------------

def create_tenant(name: str, llm_policy: str = "default") -> str:
    tenant_id = "t_" + secrets.token_hex(6)
    with _lock, _connect() as db:
        db.execute("INSERT INTO tenants (tenant_id, name, created_at, llm_policy) VALUES (?, ?, ?, ?)",
                   (tenant_id, name, time.time(), llm_policy))
    return tenant_id


def create_key(tenant_id: str, label: str = "default", role: str = "member") -> str:
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    with _lock, _connect() as db:
        if not db.execute("SELECT 1 FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone():
            raise ValueError(f"Unknown tenant {tenant_id}")
        db.execute("INSERT INTO api_keys (key_hash, key_prefix, tenant_id, label, role, created_at) "
                   "VALUES (?, ?, ?, ?, ?, ?)", (_hash(key), key[:12], tenant_id, label, role, time.time()))
    return key


def revoke_key(prefix: str) -> int:
    with _lock, _connect() as db:
        return db.execute("UPDATE api_keys SET revoked = 1 WHERE key_prefix LIKE ?", (prefix + "%",)).rowcount


def set_llm_policy(tenant_id: str, policy: str) -> None:
    if policy not in ("default", "none"):
        raise ValueError("llm_policy must be 'default' or 'none'")
    with _lock, _connect() as db:
        db.execute("UPDATE tenants SET llm_policy = ? WHERE tenant_id = ?", (policy, tenant_id))


def delete_tenant(tenant_id: str) -> dict:
    """Remove a tenant and everything stored for it: jobs (transcripts,
    documents), screenshots, cached LLM responses and API keys."""
    import pipeline
    from llm import cache

    job_ids = [j["job_id"] for j in pipeline.JOB_STORE.list(tenant_id, limit=1_000_000)]
    for job_id in job_ids:
        pipeline.delete_job_data(job_id)
    cached = cache.purge_tenant(tenant_id)
    with _lock, _connect() as db:
        keys = db.execute("DELETE FROM api_keys WHERE tenant_id = ?", (tenant_id,)).rowcount
        db.execute("DELETE FROM tenants WHERE tenant_id = ?", (tenant_id,))
    return {"tenant_id": tenant_id, "jobs": len(job_ids), "cached_llm_responses": cached, "api_keys": keys}


def tenant_name(tenant_id: str) -> str:
    try:
        with _connect() as db:
            row = db.execute("SELECT name FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
        return row[0] if row else tenant_id
    except Exception:
        return tenant_id


def tenant_llm_policy(tenant_id: str) -> str:
    """'none' when the tenant does not allow external LLM calls."""
    try:
        with _connect() as db:
            row = db.execute("SELECT llm_policy FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
        return row[0] if row else "default"
    except Exception:
        return "default"


def principal_for_key(key: str) -> Optional[Principal]:
    if not key or not key.startswith(KEY_PREFIX):
        return None
    with _connect() as db:
        row = db.execute(
            "SELECT k.tenant_id, k.role, k.label, t.llm_policy FROM api_keys k JOIN tenants t USING (tenant_id) "
            "WHERE k.key_hash = ? AND k.revoked = 0", (_hash(key),)).fetchone()
    return Principal(row[0], row[1], row[2] or "", row[3]) if row else None


# --------------------------------------------------------------------------
# Request authentication
# --------------------------------------------------------------------------

LOCAL_PRINCIPAL = Principal("local", "admin", "auth-disabled")

# Browser sign-in: the API key is exchanged once for an HttpOnly session
# cookie, so the web UI never keeps the key itself (a page load cannot carry
# an Authorization header, which is why the UI shell used to open for anyone).
SESSION_COOKIE = "continuum_session"
SESSION_TTL_SECONDS = int(os.getenv("CONTINUUM_SESSION_TTL", str(12 * 3600)))
# Cookie-authenticated requests that change data must carry this header. A
# cross-site form or image cannot add it, so the cookie alone is never enough
# to make a change (CSRF).
CSRF_HEADER = "x-continuum-csrf"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def create_session(principal: Principal) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _lock, _connect() as db:
        db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        db.execute("INSERT INTO sessions (session_hash, tenant_id, key_label, role, created_at, expires_at) "
                   "VALUES (?, ?, ?, ?, ?, ?)",
                   (_hash(token), principal.tenant_id, principal.key_label, principal.role, now, now + SESSION_TTL_SECONDS))
    return token


def principal_for_session(token: Optional[str]) -> Optional[Principal]:
    if not token:
        return None
    with _connect() as db:
        row = db.execute(
            "SELECT s.tenant_id, s.role, s.key_label, t.llm_policy FROM sessions s JOIN tenants t USING (tenant_id) "
            "WHERE s.session_hash = ? AND s.expires_at > ?", (_hash(token), time.time())).fetchone()
    return Principal(row[0], row[1], row[2] or "", row[3]) if row else None


def end_session(token: Optional[str]) -> None:
    if token:
        with _lock, _connect() as db:
            db.execute("DELETE FROM sessions WHERE session_hash = ?", (_hash(token),))


def optional_principal(request: Request) -> Optional[Principal]:
    """The caller's tenant if the request is authenticated (key or session),
    else None. Used where an anonymous request gets a different answer
    rather than a 401 (the UI shell shows the sign-in page)."""
    if not auth_enabled():
        return LOCAL_PRINCIPAL
    header = request.headers.get("authorization", "")
    key = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "").strip()
    if key:
        return principal_for_key(key)
    return principal_for_session(request.cookies.get(SESSION_COOKIE))


def require_principal(request: Request) -> Principal:
    """FastAPI dependency: the tenant a request acts for, or 401."""
    global _warned_disabled
    if not auth_enabled():
        if not _warned_disabled:
            logger.warning("CONTINUUM_AUTH=disabled: every request acts as tenant 'local'. Never use this in production.")
            _warned_disabled = True
        return LOCAL_PRINCIPAL
    header = request.headers.get("authorization", "")
    key = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "").strip()
    if key:
        principal = principal_for_key(key)
    else:
        principal = principal_for_session(request.cookies.get(SESSION_COOKIE))
        if principal is not None and request.method not in _SAFE_METHODS and request.headers.get(CSRF_HEADER) != "1":
            raise HTTPException(status_code=403, detail="Missing CSRF header.")
    if principal is None:
        raise HTTPException(status_code=401, detail="Sign in or provide a valid API key.",
                            headers={"WWW-Authenticate": "Bearer"})
    return principal


def ensure_job_access(job: Optional[dict], principal: Principal) -> dict:
    """The job, if it belongs to the caller's tenant; otherwise 404 (the
    same answer as a job that does not exist)."""
    if not job or (job.get("tenant_id") or "local") != principal.tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# --------------------------------------------------------------------------
# Signed screenshot URLs (an <img> tag cannot send an Authorization header)
# --------------------------------------------------------------------------

def _secret() -> bytes:
    env = os.getenv("CONTINUUM_SECRET_KEY")
    if env:
        return env.encode("utf-8")
    with _lock, _connect() as db:
        row = db.execute("SELECT value FROM settings WHERE name = 'asset_signing_key'").fetchone()
        if row:
            return row[0].encode("utf-8")
        value = secrets.token_hex(32)
        db.execute("INSERT INTO settings (name, value) VALUES ('asset_signing_key', ?)", (value,))
        return value.encode("utf-8")


def _signature(path: str, expires: int) -> str:
    return hmac.new(_secret(), f"{path}|{expires}".encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def sign_asset_path(path: str) -> str:
    expires = int(time.time()) + ASSET_URL_TTL_SECONDS
    return f"{path}?exp={expires}&sig={_signature(path, expires)}"


def verify_asset_signature(path: str, exp: Optional[int], sig: Optional[str]) -> bool:
    if not exp or not sig or exp < time.time():
        return False
    return hmac.compare_digest(_signature(path, int(exp)), sig)


def sign_asset_urls(value):
    """Return a copy of a job payload with every /kt-assets/ path signed."""
    if isinstance(value, dict):
        return {k: sign_asset_urls(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sign_asset_urls(v) for v in value]
    if isinstance(value, str) and value.startswith("/kt-assets/") and "?" not in value:
        return sign_asset_path(value)
    return value


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "create-tenant" and args:
        tenant_id = create_tenant(" ".join(args))
        key = create_key(tenant_id, "first key", role="admin")
        print(f"tenant: {tenant_id}\napi key (shown once, store it securely): {key}")
    elif cmd == "create-key" and args:
        print(create_key(args[0], " ".join(args[1:]) or "key"))
    elif cmd == "revoke" and args:
        print(f"revoked {revoke_key(args[0])} key(s)")
    elif cmd == "list":
        with _connect() as db:
            for t in db.execute("SELECT tenant_id, name, llm_policy FROM tenants ORDER BY created_at"):
                keys = db.execute("SELECT key_prefix, label, role, revoked FROM api_keys WHERE tenant_id = ?",
                                  (t[0],)).fetchall()
                print(f"{t[0]}  {t[1]}  (llm: {t[2]})")
                for k in keys:
                    print(f"    {k[0]}…  {k[1]}  {k[2]}{'  REVOKED' if k[3] else ''}")
    elif cmd == "delete-tenant" and args:
        print(delete_tenant(args[0]))
    elif cmd == "llm-policy" and len(args) == 2:
        set_llm_policy(args[0], args[1])
        print("ok")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
