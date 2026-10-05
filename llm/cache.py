"""Persistent exact-response cache for LLM calls.

The key is a hash of everything that determines the request: provider, model,
temperature, output budget, stop sequences, system prompt and the full prompt
text (which already contains every transcript input). A changed prompt
template, a changed input or a changed model therefore produces a different
key, so a stored response is only ever reused for the identical request. Each
call site re-applies its own validation to the response, exactly as on the
first run, so a cache hit reproduces that run's outcome.

Stored: the key hash and the response text. Prompts are not stored.

Settings (environment):
  LLM_CACHE            readwrite (default) | read | off
  LLM_CACHE_PATH       default data/llm_cache.sqlite
  LLM_CACHE_TTL_DAYS   default 90 (0 = never expire)
  LLM_CACHE_NAMESPACE  optional; change it to invalidate everything, or set
                       one per tenant so tenants never share entries
"""
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional

from llm.tenant_context import CURRENT_TENANT

LOGGER = logging.getLogger(__name__)

# Bump when the key layout or stored format changes.
CACHE_SCHEMA_VERSION = "1"

_INIT_LOCK = threading.Lock()
_INITIALISED: set = set()


def cache_mode() -> str:
    mode = os.getenv("LLM_CACHE", "readwrite").strip().lower()
    return mode if mode in ("readwrite", "read", "off") else "readwrite"


def cache_path() -> str:
    # Absolute, so the cache does not move with the working directory.
    return os.getenv("LLM_CACHE_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "data", "llm_cache.sqlite"))


def _ttl_seconds() -> float:
    try:
        days = float(os.getenv("LLM_CACHE_TTL_DAYS", "90"))
    except ValueError:
        days = 90.0
    return days * 86400.0 if days > 0 else 0.0


def make_key(*, provider: str, model: str, prompt: str, params: Dict[str, Any]) -> str:
    material = json.dumps(
        {
            "v": CACHE_SCHEMA_VERSION,
            "ns": os.getenv("LLM_CACHE_NAMESPACE", ""),
            # Tenants never share entries: identical prompts from two
            # customers are two cache rows, each deletable with its tenant.
            "tenant": CURRENT_TENANT.get(),
            "provider": provider,
            "model": model,
            "params": params,
            "prompt": prompt,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _connect(path: str) -> sqlite3.Connection:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    if path not in _INITIALISED:
        with _INIT_LOCK:
            if path not in _INITIALISED:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS llm_responses ("
                    " key TEXT PRIMARY KEY, call_site TEXT, provider TEXT, model TEXT,"
                    " response TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER,"
                    " created_at REAL, last_hit_at REAL, hit_count INTEGER DEFAULT 0)"
                )
                try:
                    conn.execute("ALTER TABLE llm_responses ADD COLUMN tenant TEXT")
                except sqlite3.OperationalError:
                    pass  # column already present
                conn.commit()
                _INITIALISED.add(path)
    return conn


def get(key: str) -> Optional[Dict[str, Any]]:
    if cache_mode() == "off":
        return None
    try:
        conn = _connect(cache_path())
        try:
            row = conn.execute(
                "SELECT response, input_tokens, output_tokens, created_at FROM llm_responses WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            ttl = _ttl_seconds()
            if ttl and row[3] and time.time() - row[3] > ttl:
                return None
            if cache_mode() == "readwrite":
                conn.execute(
                    "UPDATE llm_responses SET hit_count = hit_count + 1, last_hit_at = ? WHERE key = ?",
                    (time.time(), key),
                )
                conn.commit()
            return {"response": row[0], "input_tokens": row[1] or 0, "output_tokens": row[2] or 0}
        finally:
            conn.close()
    except Exception as exc:  # a cache problem must never fail a KT
        LOGGER.warning("LLM cache read failed: %s", exc)
        return None


def put(key: str, *, response: str, call_site: str, provider: str, model: str,
        input_tokens: int, output_tokens: int) -> None:
    if cache_mode() != "readwrite" or not isinstance(response, str) or not response.strip():
        return
    try:
        conn = _connect(cache_path())
        try:
            conn.execute(
                "INSERT OR REPLACE INTO llm_responses"
                " (key, call_site, provider, model, response, input_tokens, output_tokens, created_at, last_hit_at, hit_count, tenant)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?)",
                (key, call_site, provider, model, response, int(input_tokens or 0), int(output_tokens or 0), time.time(),
                 CURRENT_TENANT.get()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        LOGGER.warning("LLM cache write failed: %s", exc)


def purge(*, older_than_days: Optional[float] = None) -> int:
    """Delete cached responses (all of them, or those older than N days)."""
    conn = _connect(cache_path())
    try:
        if older_than_days is None:
            cur = conn.execute("DELETE FROM llm_responses")
        else:
            cur = conn.execute(
                "DELETE FROM llm_responses WHERE created_at < ?",
                (time.time() - older_than_days * 86400.0,),
            )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()



def purge_tenant(tenant_id: str) -> int:
    """Delete every cached response produced for one tenant."""
    conn = _connect(cache_path())
    try:
        cur = conn.execute("DELETE FROM llm_responses WHERE tenant = ?", (tenant_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    days = float(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"purged {purge(older_than_days=days)} cached LLM responses from {cache_path()}")
