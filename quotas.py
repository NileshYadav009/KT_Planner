"""Daily limits per workspace (the due-diligence review, P1-3).

A deployment's workers and its LLM account are shared by all of its
workspaces, so one workspace's batch could hold up everyone else's KTs
and use up the day's LLM quota. The queue takes turns between workspaces
(job_queue.TaskQueue.claim); this module limits how much each workspace
uses in the last 24 hours:

    kts      KTs and follow-up sessions submitted       CONTINUUM_KTS_PER_DAY
    minutes  minutes of recording submitted              CONTINUUM_RECORDING_MINUTES_PER_DAY
    tokens   LLM tokens (sent and received) of its KTs   CONTINUUM_LLM_TOKENS_PER_DAY

0 or unset means no limit, the default. A workspace's own limit replaces
the deployment's:

    python -m auth limits <tenant_id>
    python -m auth limits-set <tenant_id> [--kts N] [--minutes N] [--tokens N]   # "default": the deployment's

A submission over the KT limit is refused (HTTP 429, with Retry-After)
and nothing is queued; the message says which limit was reached and when
the next submission fits. A recording's minutes are known only once a
worker has probed it (the API runs no media tool, P1-4): over the minutes
limit, its KT (or session) fails with the same kind of message, and the KT
it was counted as is taken back. Each check and its record are one
transaction, so two submissions at the same moment cannot both take the
last place. The token limit is checked before each LLM call
(llm/cached_provider): once it is reached, the KT's remaining steps keep
their rules-only results, as when a per-KT budget runs out, and the
document says so. A KT's tokens are recorded when it ends, so two KTs of
one workspace running at the same time can go over the limit by what they
use between them.

Every submission, recording minute and LLM token is recorded, with or
without a limit (usage_events), so a workspace's use can be shown and
billed later. The rows hold numbers and ids only; they are removed with the
workspace.
"""
import logging
import os
import sqlite3
import time
from typing import Dict, Iterable, List, Optional, Tuple

from job_store import DEFAULT_DB_PATH, ClosingConnection

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 24 * 3600
KINDS = ("kts", "minutes", "tokens")
ENV = {"kts": "CONTINUUM_KTS_PER_DAY", "minutes": "CONTINUUM_RECORDING_MINUTES_PER_DAY",
       "tokens": "CONTINUUM_LLM_TOKENS_PER_DAY"}
_OPERATOR = "Your Continuum operator can raise the limit."

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenant_quotas (
    tenant_id   TEXT PRIMARY KEY,
    kts         REAL,
    minutes     REAL,
    tokens      REAL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    at          REAL NOT NULL,
    kind        TEXT NOT NULL,
    amount      REAL NOT NULL,
    job_id      TEXT,
    ref         TEXT
);
CREATE INDEX IF NOT EXISTS usage_events_tenant ON usage_events(tenant_id, kind, at);
"""


class LimitReached(Exception):
    """A submission the workspace's daily limit has no room for."""

    def __init__(self, kind: str, message: str, retry_after: Optional[int]):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retry_after = retry_after


_READY: set = set()


def _connect(autocommit: bool = False) -> sqlite3.Connection:
    path = os.getenv("CONTINUUM_DB_PATH", DEFAULT_DB_PATH)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    db = sqlite3.connect(path, timeout=30, factory=ClosingConnection, isolation_level=None if autocommit else "")
    if path not in _READY:
        db.executescript(_SCHEMA)
        if "ref" not in {row[1] for row in db.execute("PRAGMA table_info(usage_events)")}:
            db.execute("ALTER TABLE usage_events ADD COLUMN ref TEXT")      # a table made before 11 Nov
        _READY.add(path)
    return db


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------

def _number(value) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def deployment_limit(kind: str) -> float:
    """The deployment's limit for every workspace without its own (0: none)."""
    return _number(os.getenv(ENV[kind], "0") or 0)


def limits(tenant_id: str) -> Dict[str, float]:
    """The workspace's limits, its own where set (0: no limit)."""
    with _connect() as db:
        row = db.execute("SELECT kts, minutes, tokens FROM tenant_quotas WHERE tenant_id = ?", (tenant_id,)).fetchone()
    own = dict(zip(KINDS, row, strict=True)) if row else {}
    return {kind: _number(own[kind]) if own.get(kind) is not None else deployment_limit(kind) for kind in KINDS}


def own_limits(tenant_id: str) -> Dict[str, Optional[float]]:
    """The limits set for this workspace itself (None: the deployment's)."""
    with _connect() as db:
        row = db.execute("SELECT kts, minutes, tokens FROM tenant_quotas WHERE tenant_id = ?", (tenant_id,)).fetchone()
    return dict(zip(KINDS, row, strict=True)) if row else {kind: None for kind in KINDS}


def set_limits(tenant_id: str, **values) -> Dict[str, Optional[float]]:
    """Set this workspace's own limits: a number (0: no limit) or None (the
    deployment's). Kinds not given keep their setting."""
    unknown = set(values) - set(KINDS)
    if unknown:
        raise ValueError(f"Unknown limit {sorted(unknown)[0]!r}; use {', '.join(KINDS)}")
    for kind, value in values.items():
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise ValueError(f"The {kind} limit must be 0 or more")
    merged = dict(own_limits(tenant_id), **values)
    with _connect() as db:
        if all(merged[kind] is None for kind in KINDS):
            db.execute("DELETE FROM tenant_quotas WHERE tenant_id = ?", (tenant_id,))
        else:
            db.execute("INSERT INTO tenant_quotas (tenant_id, kts, minutes, tokens, updated_at) VALUES (?, ?, ?, ?, ?) "
                       "ON CONFLICT(tenant_id) DO UPDATE SET kts = excluded.kts, minutes = excluded.minutes, "
                       "tokens = excluded.tokens, updated_at = excluded.updated_at",
                       (tenant_id, merged["kts"], merged["minutes"], merged["tokens"], time.time()))
    return merged


# --------------------------------------------------------------------------
# Use
# --------------------------------------------------------------------------

def used(tenant_id: str, now: Optional[float] = None) -> Dict[str, float]:
    """What the workspace used in the last 24 hours."""
    now = time.time() if now is None else now
    with _connect() as db:
        rows = db.execute("SELECT kind, SUM(amount) FROM usage_events WHERE tenant_id = ? AND at > ? GROUP BY kind",
                          (tenant_id, now - WINDOW_SECONDS)).fetchall()
    totals = {kind: 0.0 for kind in KINDS}
    totals.update({kind: float(total or 0) for kind, total in rows if kind in totals})
    return totals


def usage(tenant_id: str, now: Optional[float] = None) -> Dict[str, object]:
    """Limits and use in the last 24 hours, for GET /usage and the CLI."""
    lim, use = limits(tenant_id), used(tenant_id, now)
    return {"window_hours": WINDOW_SECONDS // 3600,
            "kts": {"used": int(use["kts"]), "limit": int(lim["kts"]) or None},
            "recording_minutes": {"used": round(use["minutes"], 1), "limit": lim["minutes"] or None},
            "llm_tokens": {"used": int(use["tokens"]), "limit": int(lim["tokens"]) or None}}


def _duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "under a minute"
    hours, minutes = divmod(minutes, 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def _when(retry_after: int, now: float) -> str:
    at = time.strftime("%H:%M UTC", time.gmtime(now + retry_after))
    return f"in about {_duration(retry_after)} (at {at})"


def _minutes(value: float) -> str:
    return f"{max(1, round(value))}"


def _refusal(kind: str, limit: float, events: List[Tuple[float, float]], amount: float,
             now: float) -> Optional[LimitReached]:
    """None when `amount` fits under `limit` beside `events` (the window's
    (time, amount) rows, oldest first); otherwise the refusal, with the time
    the oldest rows leave the window and make room."""
    total = sum(a for _, a in events)
    if total + amount <= limit:
        return None
    if kind == "minutes" and amount > limit:
        return LimitReached(kind, f"This recording is {_minutes(amount)} minutes long, more than this workspace's "
                                  f"daily limit of {limit:g} recording minutes. {_OPERATOR}", None)
    retry_after = None
    freed = 0.0
    for at, a in events:
        freed += a
        if total - freed + amount <= limit:
            retry_after = max(1, int(at + WINDOW_SECONDS - now + 1))
            break
    when = _when(retry_after, now) if retry_after else "later"
    if kind == "kts":
        message = (f"This workspace has submitted {total:g} KTs and sessions in the last 24 hours, its daily "
                   f"limit of {limit:g}. The next one can be submitted {when}. {_OPERATOR}")
    else:
        message = (f"This recording is {_minutes(amount)} minutes long, and this workspace has used "
                   f"{_minutes(total)} of its {limit:g} recording minutes in the last 24 hours. It can be "
                   f"submitted {when}. {_OPERATOR}")
    return LimitReached(kind, message, retry_after)


def admit(tenant_id: str, job_id: str, minutes: Optional[float] = None, now: Optional[float] = None,
          count_kt: bool = True, ref: Optional[str] = None) -> List[int]:
    """Check a submission (one KT or session, and its recording's minutes)
    against the workspace's limits and record it, in one transaction.
    Returns the rows recorded, for release() if it is not queued after all.
    Raises LimitReached, recording nothing. `count_kt=False` checks and
    records only the minutes (a recording a worker has just probed). With
    `ref`, what was already recorded under it is not counted again (a KT's
    second attempt after its worker died)."""
    now = time.time() if now is None else now
    lim = limits(tenant_id)
    wanted = _wanted(minutes, count_kt)
    db = _connect(autocommit=True)
    try:
        db.execute("BEGIN IMMEDIATE")
        try:
            if ref is not None and db.execute("SELECT 1 FROM usage_events WHERE tenant_id = ? AND ref = ? LIMIT 1",
                                              (tenant_id, ref)).fetchone():
                wanted = []
            for kind, amount in wanted:
                if lim[kind]:
                    refusal = _refusal(kind, lim[kind], _events(db, tenant_id, kind, now), amount, now)
                    if refusal:
                        raise refusal
            rows = [db.execute("INSERT INTO usage_events (tenant_id, at, kind, amount, job_id, ref) "
                               "VALUES (?, ?, ?, ?, ?, ?)", (tenant_id, now, kind, amount, job_id, ref)).lastrowid
                    for kind, amount in wanted]
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    finally:
        db.close()
    return rows


def release(rows: Iterable[int]) -> None:
    """Take back a submission admitted but not queued (an error after admit)."""
    rows = [int(r) for r in rows or []]
    if rows:
        try:
            with _connect() as db:
                db.execute("DELETE FROM usage_events WHERE id IN (%s)" % ",".join("?" * len(rows)), rows)
        except sqlite3.Error:      # never in place of the error that brought us here
            logger.warning("Could not take back usage rows %s", rows, exc_info=True)


def _wanted(minutes: Optional[float], count_kt: bool = True) -> List[Tuple[str, float]]:
    wanted = [("kts", 1.0)] if count_kt else []
    if minutes:
        wanted.append(("minutes", float(minutes)))
    return wanted


def _events(db: sqlite3.Connection, tenant_id: str, kind: str, now: float) -> List[Tuple[float, float]]:
    return db.execute("SELECT at, amount FROM usage_events WHERE tenant_id = ? AND kind = ? AND at > ? ORDER BY at",
                      (tenant_id, kind, now - WINDOW_SECONDS)).fetchall()


# --------------------------------------------------------------------------
# LLM tokens
# --------------------------------------------------------------------------

def tokens_left(tenant_id: str, now: Optional[float] = None) -> Optional[float]:
    """LLM tokens the workspace may still use today (None: no limit). Read
    when a KT starts; the KT's own calls are counted against it as they go."""
    limit = limits(tenant_id)["tokens"]
    if not limit:
        return None
    return limit - used(tenant_id, now)["tokens"]


def record_tokens(tenant_id: str, job_id: str, tokens: float, now: Optional[float] = None) -> None:
    """A KT's LLM tokens, recorded when it ends."""
    if tokens and tokens > 0:
        with _connect() as db:
            db.execute("INSERT INTO usage_events (tenant_id, at, kind, amount, job_id) VALUES (?, ?, 'tokens', ?, ?)",
                       (tenant_id, time.time() if now is None else now, float(tokens), job_id))


def token_limit_notice(tenant_id: str, skipped: int) -> str:
    limit = limits(tenant_id)["tokens"]
    return (f"This workspace reached its daily LLM limit ({limit:,.0f} tokens in 24 hours): {skipped} LLM step(s) "
            "of this KT kept their rules-only result.")


def delete_tenant(tenant_id: str) -> int:
    """A deleted workspace's limits and recorded use (auth.delete_tenant)."""
    with _connect() as db:
        db.execute("DELETE FROM tenant_quotas WHERE tenant_id = ?", (tenant_id,))
        return db.execute("DELETE FROM usage_events WHERE tenant_id = ?", (tenant_id,)).rowcount
