"""Tenants, people, roles and request authentication (P0-1).

Before this module every route was public: any caller could read, change or
export any job, and /status returned the full transcript to anyone with a
job id.

Model: a tenant (a customer workspace) has people and API keys. People sign
in through the workspace's identity provider (sso.py, OpenID Connect) or, for
small pilots, with an API key; machines use API keys. Every job belongs to the
tenant that created it and is invisible to every other tenant (404, so
existence is not revealed). Keys are shown once and stored only as SHA-256
hashes.

Roles (what each may do is in PERMISSIONS):
    admin     everything, including deleting any KT and managing people
    reviewer  create KTs, correct them, delete their own
    giver     the person handing over: create KTs, correct them, delete their own
    receiver  the person taking over: read and export only

    python -m auth create-tenant "Acme Transition Team"   # prints tenant id + first (admin) key
    python -m auth create-key <tenant_id> [label] [--role giver]
    python -m auth list
    python -m auth revoke <key prefix>
    python -m auth llm-policy <tenant_id> none|default   # "none": no external LLM
    python -m auth delete-tenant <tenant_id>            # jobs, screenshots, cache, keys, people

    python -m auth add-user <tenant_id> <email> <role>   # pre-approve a person (before their first SSO sign-in)
    python -m auth set-role <tenant_id> <email> <role>
    python -m auth disable-user <tenant_id> <email>      # ends their sessions at once
    python -m auth enable-user <tenant_id> <email>
    python -m auth users <tenant_id>
    python -m auth audit <tenant_id> [count]            # sign-ins, role changes, deletions

    python -m auth sso-set <tenant_id> --issuer URL --client-id ID --domains acme.com[,acme.co.uk]
        [--client-secret-env VAR] [--default-role receiver] [--invite-only]
        [--role-claim roles --role-map "Continuum.Admin=admin,Continuum.Reviewer=reviewer"] [--enforce]
    python -m auth sso-show <tenant_id>
    python -m auth sso-remove <tenant_id>

CONTINUUM_AUTH=disabled turns authentication off for local development only
(every request then acts as tenant "local"); it is logged loudly.
"""
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request

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
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    email       TEXT NOT NULL,
    name        TEXT,
    role        TEXT NOT NULL,
    issuer      TEXT,
    subject     TEXT,
    created_at  REAL NOT NULL,
    last_login  REAL,
    disabled    INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS users_tenant_email ON users(tenant_id, email);
CREATE UNIQUE INDEX IF NOT EXISTS users_identity ON users(tenant_id, issuer, subject);
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          REAL NOT NULL,
    tenant_id   TEXT,
    actor       TEXT,
    action      TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS audit_tenant_at ON audit_log(tenant_id, at DESC);
CREATE TABLE IF NOT EXISTS sso_providers (
    tenant_id         TEXT PRIMARY KEY,
    issuer            TEXT NOT NULL,
    client_id         TEXT NOT NULL,
    client_secret_env TEXT,
    domains           TEXT NOT NULL,
    default_role      TEXT NOT NULL DEFAULT 'receiver',
    invite_only       INTEGER NOT NULL DEFAULT 0,
    role_claim        TEXT,
    role_map          TEXT,
    enforce           INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sso_requests (
    state_hash    TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    nonce         TEXT NOT NULL,
    verifier      TEXT NOT NULL,
    redirect_uri  TEXT NOT NULL,
    next_path     TEXT,
    expires_at    REAL NOT NULL
);
"""

# Columns added after the first release; older databases are upgraded in place.
_MIGRATIONS = (
    ("sessions", "user_id", "ALTER TABLE sessions ADD COLUMN user_id TEXT"),
)

_warned_disabled = False
_lock = threading.Lock()
_migrated: set = set()

# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------

ROLES = ("admin", "reviewer", "giver", "receiver")
ROLE_LABELS = {"admin": "Admin", "reviewer": "Reviewer", "giver": "KT giver", "receiver": "KT receiver"}
# Keys created before roles existed were "member": they could create and
# correct KTs, which is what a giver does.
_LEGACY_ROLES = {"member": "giver"}
PERMISSIONS = {
    "admin": {"kt:read", "kt:create", "kt:correct", "kt:delete_own", "kt:delete_any", "people:manage"},
    "reviewer": {"kt:read", "kt:create", "kt:correct", "kt:delete_own"},
    "giver": {"kt:read", "kt:create", "kt:correct", "kt:delete_own"},
    "receiver": {"kt:read"},
}
_ROLE_RANK = {role: rank for rank, role in enumerate(reversed(ROLES))}   # receiver 0 … admin 3


def normalise_role(role: Optional[str]) -> str:
    role = (role or "").strip().lower()
    role = _LEGACY_ROLES.get(role, role)
    if role not in ROLES:
        raise ValueError(f"Unknown role '{role}'. Use one of: {', '.join(ROLES)}")
    return role


def highest_role(roles) -> Optional[str]:
    ranked = [r for r in roles if r in _ROLE_RANK]
    return max(ranked, key=_ROLE_RANK.get) if ranked else None


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    role: str
    key_label: str             # who: a person's email, or an API key's label
    llm_policy: str = "default"
    user_id: Optional[str] = None   # "u_…" for a person, "key:<prefix>" for an API key
    method: str = "key"        # "sso", "key" or "disabled"

    @property
    def actor(self) -> str:
        return self.user_id or f"key:{self.key_label}"

    @property
    def permissions(self) -> set:
        try:
            return PERMISSIONS[normalise_role(self.role)]
        except ValueError:
            return set()

    def can(self, permission: str) -> bool:
        return permission in self.permissions


def _db_path() -> str:
    return os.getenv("CONTINUUM_DB_PATH", DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.executescript(_SCHEMA)
    if path not in _migrated:
        for table, column, sql in _MIGRATIONS:
            if column not in {row[1] for row in db.execute(f"PRAGMA table_info({table})")}:
                db.execute(sql)
        db.commit()
        _migrated.add(path)
    return db


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def auth_enabled() -> bool:
    return os.getenv("CONTINUUM_AUTH", "required").strip().lower() not in ("disabled", "off", "0", "false")


def audit(tenant_id: Optional[str], actor: Optional[str], action: str, detail: Optional[dict] = None) -> None:
    """Record a security-relevant event (sign-in, role change, deletion).
    Never pass secrets or transcript text in detail."""
    try:
        with _lock, _connect() as db:
            db.execute("INSERT INTO audit_log (at, tenant_id, actor, action, detail) VALUES (?, ?, ?, ?, ?)",
                       (time.time(), tenant_id, actor, action, json.dumps(detail or {}, default=str)))
    except Exception:
        logger.exception("Could not write audit event %s", action)


def audit_events(tenant_id: str, limit: int = 100) -> list:
    with _connect() as db:
        rows = db.execute("SELECT at, actor, action, detail FROM audit_log WHERE tenant_id = ? "
                          "ORDER BY at DESC, id DESC LIMIT ?", (tenant_id, int(limit))).fetchall()
    return [{"at": r[0], "actor": r[1], "action": r[2], "detail": json.loads(r[3] or "{}")} for r in rows]


# --------------------------------------------------------------------------
# Administration
# --------------------------------------------------------------------------

def create_tenant(name: str, llm_policy: str = "default") -> str:
    tenant_id = "t_" + secrets.token_hex(6)
    with _lock, _connect() as db:
        db.execute("INSERT INTO tenants (tenant_id, name, created_at, llm_policy) VALUES (?, ?, ?, ?)",
                   (tenant_id, name, time.time(), llm_policy))
    return tenant_id


def _require_tenant(db, tenant_id: str) -> None:
    if not db.execute("SELECT 1 FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone():
        raise ValueError(f"Unknown tenant {tenant_id}")


def create_key(tenant_id: str, label: str = "default", role: str = "giver") -> str:
    role = normalise_role(role)
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    with _lock, _connect() as db:
        _require_tenant(db, tenant_id)
        db.execute("INSERT INTO api_keys (key_hash, key_prefix, tenant_id, label, role, created_at) "
                   "VALUES (?, ?, ?, ?, ?, ?)", (_hash(key), key[:12], tenant_id, label, role, time.time()))
    return key


def revoke_key(prefix: str) -> int:
    """Revoke keys by prefix. Browser sessions opened with those keys end too."""
    with _lock, _connect() as db:
        prefixes = [r[0] for r in db.execute("SELECT key_prefix FROM api_keys WHERE key_prefix LIKE ? AND revoked = 0",
                                             (prefix + "%",)).fetchall()]
        count = db.execute("UPDATE api_keys SET revoked = 1 WHERE key_prefix LIKE ?", (prefix + "%",)).rowcount
        for p in prefixes:
            db.execute("DELETE FROM sessions WHERE user_id = ?", (f"key:{p}",))
    return count


def set_llm_policy(tenant_id: str, policy: str) -> None:
    if policy not in ("default", "none"):
        raise ValueError("llm_policy must be 'default' or 'none'")
    with _lock, _connect() as db:
        db.execute("UPDATE tenants SET llm_policy = ? WHERE tenant_id = ?", (policy, tenant_id))


def delete_tenant(tenant_id: str) -> dict:
    """Remove a tenant and everything stored for it: jobs (transcripts,
    documents), screenshots, cached LLM responses, API keys, people, sessions
    and its SSO configuration."""
    import pipeline
    from llm import cache

    job_ids = [j["job_id"] for j in pipeline.JOB_STORE.list(tenant_id, limit=1_000_000)]
    for job_id in job_ids:
        pipeline.delete_job_data(job_id)
    cached = cache.purge_tenant(tenant_id)
    with _lock, _connect() as db:
        keys = db.execute("DELETE FROM api_keys WHERE tenant_id = ?", (tenant_id,)).rowcount
        people = db.execute("DELETE FROM users WHERE tenant_id = ?", (tenant_id,)).rowcount
        db.execute("DELETE FROM sessions WHERE tenant_id = ?", (tenant_id,))
        db.execute("DELETE FROM sso_providers WHERE tenant_id = ?", (tenant_id,))
        db.execute("DELETE FROM sso_requests WHERE tenant_id = ?", (tenant_id,))
        db.execute("DELETE FROM tenants WHERE tenant_id = ?", (tenant_id,))
    audit(tenant_id, "cli", "tenant_deleted", {"jobs": len(job_ids), "people": people, "api_keys": keys})
    return {"tenant_id": tenant_id, "jobs": len(job_ids), "cached_llm_responses": cached, "api_keys": keys,
            "people": people}


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
            "SELECT k.tenant_id, k.role, k.label, t.llm_policy, k.key_prefix FROM api_keys k "
            "JOIN tenants t USING (tenant_id) WHERE k.key_hash = ? AND k.revoked = 0", (_hash(key),)).fetchone()
    if not row:
        return None
    return Principal(row[0], _LEGACY_ROLES.get(row[1], row[1]), row[2] or "", row[3], user_id=f"key:{row[4]}",
                     method="key")


# --------------------------------------------------------------------------
# People (signed in through SSO, see sso.py)
# --------------------------------------------------------------------------

def _user_row(db, tenant_id: str, email: str):
    return db.execute("SELECT user_id, role, disabled, issuer, subject FROM users WHERE tenant_id = ? AND email = ?",
                      (tenant_id, email.strip().lower())).fetchone()


def add_user(tenant_id: str, email: str, role: str, name: Optional[str] = None) -> str:
    """Pre-approve a person. Their identity provider account is bound to this
    record at their first sign-in."""
    role = normalise_role(role)
    email = email.strip().lower()
    with _lock, _connect() as db:
        _require_tenant(db, tenant_id)
        if _user_row(db, tenant_id, email):
            raise ValueError(f"{email} is already in this workspace")
        user_id = "u_" + secrets.token_hex(8)
        db.execute("INSERT INTO users (user_id, tenant_id, email, name, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (user_id, tenant_id, email, name, role, time.time()))
    audit(tenant_id, "cli", "user_added", {"email": email, "role": role})
    return user_id


def set_user_role(tenant_id: str, email: str, role: str, actor: str = "cli") -> None:
    role = normalise_role(role)
    with _lock, _connect() as db:
        row = _user_row(db, tenant_id, email)
        if not row:
            raise ValueError(f"No person {email} in this workspace")
        db.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, row[0]))
    audit(tenant_id, actor, "role_changed", {"email": email.strip().lower(), "from": row[1], "to": role})


def set_user_disabled(tenant_id: str, email: str, disabled: bool, actor: str = "cli") -> None:
    """Turning a person off ends every session they have, immediately."""
    with _lock, _connect() as db:
        row = _user_row(db, tenant_id, email)
        if not row:
            raise ValueError(f"No person {email} in this workspace")
        db.execute("UPDATE users SET disabled = ? WHERE user_id = ?", (1 if disabled else 0, row[0]))
        if disabled:
            db.execute("DELETE FROM sessions WHERE user_id = ?", (row[0],))
    audit(tenant_id, actor, "user_disabled" if disabled else "user_enabled", {"email": email.strip().lower()})


def list_users(tenant_id: str) -> list:
    with _connect() as db:
        rows = db.execute("SELECT email, name, role, disabled, last_login, subject IS NOT NULL FROM users "
                          "WHERE tenant_id = ? ORDER BY email", (tenant_id,)).fetchall()
    return [{"email": r[0], "name": r[1], "role": r[2], "disabled": bool(r[3]), "last_login": r[4],
             "signed_in_before": bool(r[5])} for r in rows]


def sso_sign_in(tenant_id: str, issuer: str, subject: str, email: str, name: Optional[str],
                role_from_idp: Optional[str], default_role: str, invite_only: bool) -> dict:
    """Find or create the person for a verified ID token. Returns the user
    record, or raises PermissionError with a reason code.

    The person is matched on the token's issuer and subject. The email is
    used once, to bind a pre-approved person on their first sign-in, and is
    otherwise only a display name and a domain check."""
    email = email.strip().lower()
    now = time.time()
    with _lock, _connect() as db:
        row = db.execute("SELECT user_id, role, disabled, email FROM users WHERE tenant_id = ? AND issuer = ? "
                         "AND subject = ?", (tenant_id, issuer, subject)).fetchone()
        if row is None:
            pre = _user_row(db, tenant_id, email)
            if pre is not None and pre[4] is None:            # pre-approved, first sign-in: bind the identity
                db.execute("UPDATE users SET issuer = ?, subject = ? WHERE user_id = ?", (issuer, subject, pre[0]))
                row = (pre[0], pre[1], pre[2], email)
            elif pre is not None:
                # The email belongs to a different identity in this workspace.
                raise PermissionError("sso_not_allowed")
            elif invite_only:
                raise PermissionError("sso_not_allowed")
            else:
                user_id = "u_" + secrets.token_hex(8)
                role = role_from_idp or normalise_role(default_role)
                db.execute("INSERT INTO users (user_id, tenant_id, email, name, role, issuer, subject, created_at) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (user_id, tenant_id, email, name, role, issuer, subject, now))
                row = (user_id, role, 0, email)
        user_id, role, disabled, stored_email = row
        if disabled:
            raise PermissionError("sso_disabled")
        if role_from_idp:                     # the IdP's groups are the source of truth when mapped
            role = role_from_idp
        # Follow an email change at the IdP, unless that address already
        # belongs to someone else in the workspace.
        if email != stored_email and _user_row(db, tenant_id, email) is None:
            stored_email = email
        db.execute("UPDATE users SET role = ?, name = COALESCE(?, name), email = ?, last_login = ? WHERE user_id = ?",
                   (role, name, stored_email, now, user_id))
    return {"user_id": user_id, "role": role, "email": stored_email, "name": name}


# --------------------------------------------------------------------------
# Request authentication
# --------------------------------------------------------------------------

LOCAL_PRINCIPAL = Principal("local", "admin", "auth-disabled", user_id="local", method="disabled")

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
        db.execute("INSERT INTO sessions (session_hash, tenant_id, key_label, role, created_at, expires_at, user_id) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (_hash(token), principal.tenant_id, principal.key_label, principal.role, now,
                    now + SESSION_TTL_SECONDS, principal.user_id))
    return token


def principal_for_session(token: Optional[str]) -> Optional[Principal]:
    """A session acts with the person's current role: a role change or a
    disabled account takes effect on the next request, not at expiry."""
    if not token:
        return None
    with _connect() as db:
        row = db.execute(
            "SELECT s.tenant_id, COALESCE(u.role, s.role), s.key_label, t.llm_policy, s.user_id, u.disabled "
            "FROM sessions s JOIN tenants t USING (tenant_id) LEFT JOIN users u ON u.user_id = s.user_id "
            "WHERE s.session_hash = ? AND s.expires_at > ?", (_hash(token), time.time())).fetchone()
    if not row or row[5]:
        return None
    user_id = row[4]
    method = "sso" if user_id and user_id.startswith("u_") else "key"
    return Principal(row[0], _LEGACY_ROLES.get(row[1], row[1]), row[2] or "", row[3], user_id=user_id, method=method)


def end_session(token: Optional[str]) -> Optional[Principal]:
    if not token:
        return None
    principal = principal_for_session(token)
    with _lock, _connect() as db:
        db.execute("DELETE FROM sessions WHERE session_hash = ?", (_hash(token),))
    return principal


def _credentials(request: Request) -> str:
    header = request.headers.get("authorization", "")
    return header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "").strip()


def optional_principal(request: Request) -> Optional[Principal]:
    """The caller's tenant if the request is authenticated (key or session),
    else None. Used where an anonymous request gets a different answer
    rather than a 401 (the UI shell shows the sign-in page)."""
    if not auth_enabled():
        return LOCAL_PRINCIPAL
    key = _credentials(request)
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
    key = _credentials(request)
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


_PERMISSION_DENIED = {
    "kt:create": "Your role can view KTs but not create them. Ask your workspace admin for the giver or reviewer role.",
    "kt:correct": "Your role can view KTs but not correct them. Ask your workspace admin for the giver or reviewer role.",
}


def require_permission(permission: str):
    """FastAPI dependency factory: the caller, if their role allows
    `permission`; otherwise 403."""
    def dependency(principal: Principal = Depends(require_principal)) -> Principal:
        if not principal.can(permission):
            raise HTTPException(status_code=403,
                                detail=_PERMISSION_DENIED.get(permission, "Your role does not allow this."))
        return principal
    return dependency


def can_delete_job(job: dict, principal: Principal) -> bool:
    """Admins may delete any KT in their workspace; givers and reviewers may
    delete the KTs they created."""
    if principal.can("kt:delete_any"):
        return True
    return principal.can("kt:delete_own") and bool(job.get("created_by_id")) and job.get("created_by_id") == principal.actor


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

def _option(args: list, name: str, default=None):
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            value = args[i + 1]
            del args[i:i + 2]
            return value
        raise SystemExit(f"{name} needs a value")
    return default


def _flag(args: list, name: str) -> bool:
    if name in args:
        args.remove(name)
        return True
    return False


def _main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], list(argv[1:])
    if cmd == "create-tenant" and args:
        tenant_id = create_tenant(" ".join(args))
        key = create_key(tenant_id, "first key", role="admin")
        print(f"tenant: {tenant_id}\napi key (shown once, store it securely): {key}")
    elif cmd == "create-key" and args:
        role = _option(args, "--role", "giver")
        print(create_key(args[0], " ".join(args[1:]) or "key", role=role))
    elif cmd == "revoke" and args:
        print(f"revoked {revoke_key(args[0])} key(s)")
    elif cmd == "list":
        with _connect() as db:
            for t in db.execute("SELECT tenant_id, name, llm_policy FROM tenants ORDER BY created_at"):
                keys = db.execute("SELECT key_prefix, label, role, revoked FROM api_keys WHERE tenant_id = ?",
                                  (t[0],)).fetchall()
                people = db.execute("SELECT COUNT(*) FROM users WHERE tenant_id = ?", (t[0],)).fetchone()[0]
                print(f"{t[0]}  {t[1]}  (llm: {t[2]}, people: {people})")
                for k in keys:
                    print(f"    {k[0]}…  {k[1]}  {_LEGACY_ROLES.get(k[2], k[2])}{'  REVOKED' if k[3] else ''}")
    elif cmd == "delete-tenant" and args:
        print(delete_tenant(args[0]))
    elif cmd == "llm-policy" and len(args) == 2:
        set_llm_policy(args[0], args[1])
        print("ok")
    elif cmd == "add-user" and len(args) == 3:
        add_user(args[0], args[1], args[2])
        print("ok")
    elif cmd == "set-role" and len(args) == 3:
        set_user_role(args[0], args[1], args[2])
        print("ok")
    elif cmd in ("disable-user", "enable-user") and len(args) == 2:
        set_user_disabled(args[0], args[1], cmd == "disable-user")
        print("ok")
    elif cmd == "users" and len(args) == 1:
        for u in list_users(args[0]):
            print(f"{u['email']:40} {u['role']:9} {'DISABLED' if u['disabled'] else ''}"
                  f"{'' if u['signed_in_before'] else ' (not signed in yet)'}")
    elif cmd == "audit" and args:
        for e in audit_events(args[0], int(args[1]) if len(args) > 1 else 50):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["at"])), e["actor"], e["action"],
                  json.dumps(e["detail"]))
    elif cmd in ("sso-set", "sso-show", "sso-remove") and args:
        import sso
        return sso.cli(cmd, args)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
