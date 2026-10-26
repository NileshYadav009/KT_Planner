"""Self-learning DevOps vocabulary: candidate detection and human-approved growth.

Detection runs automatically on every KT job (see pipeline.run_kt_pipeline) and only
ever records candidates for later review — it never changes what
devops_transcription.py corrects on its own. A human promotes a candidate into the
live, corrections-affecting vocabulary via approve_candidate() (see
scripts/review_vocabulary.py), which writes through glossary.save_glossary(). This
mirrors the "no unreviewed/hallucinated corrections" principle used throughout this
codebase's LLM-facing prompts.

Two independent heuristic signals, no ML dependency, consistent with this
codebase's existing regex-based correction style (see devops_transcription.py):
- near-miss fuzzy matches: a phrase that almost, but not quite, matches a known
  term — likely a mishearing of it, or a close variant worth reviewing
- novel technical-token patterns: a phrase that looks like DevOps terminology
  (CamelCase, short acronym, hyphenated compound, alphanumeric like "s3") but
  isn't recognized yet, especially right after a trigger phrase like "we use"
"""

import json
import logging
import os
import sqlite3
import threading
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import glossary
from devops_transcription import (
    bucket_terms_by_word_count, candidates_for_ngram, get_known_terms,
    jaro_winkler_similarity, HAS_TEXTDISTANCE,
)

logger = logging.getLogger(__name__)

CANDIDATES_PATH = os.getenv("CONTINUUM_VOCAB_CANDIDATES_PATH",
                            os.path.join(os.path.dirname(__file__), "glossary_candidates.json"))

# Matches devops_transcription.apply_fuzzy_term_corrections's auto-correct
# threshold (0.88) as the upper bound of the "near miss" gray zone. The lower
# bound is deliberately strict (not the textbook "similar enough" 0.7-0.75):
# jaro-winkler scores common short English words against short tech acronyms
# surprisingly high ("our" vs "cloudfront" ~0.77, "layer" vs "glacier" ~0.79),
# so a looser bound floods the review queue with noise. Combined with the
# stopword/length filtering below, 0.85 was the empirical point where real
# near-misses survive and generic-word noise mostly doesn't.
NEAR_MISS_LOW = 0.85
NEAR_MISS_HIGH = 0.88

TRIGGER_PHRASES = [
    "we use", "using", "deployed via", "deployed with", "built on", "powered by",
    "runs on", "hosted on", "for monitoring", "for logging", "database is",
    "pipeline", "written in", "based on",
]
TRIGGER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in TRIGGER_PHRASES) + r")\s+([a-zA-Z0-9][\w\-\.]{1,30})",
    re.IGNORECASE,
)

TECHNICAL_TOKEN_PATTERN = re.compile(
    r"^("
    r"[A-Z][a-z]+[A-Z][\w]*"       # CamelCase, e.g. CloudFront, ArgoCD
    r"|[A-Z]{2,6}"                  # short ALLCAPS acronym, e.g. VPC, SLI
    r"|[a-zA-Z]+-[a-zA-Z0-9\-]+"    # hyphenated compound, e.g. auto-scaling-group
    r"|[a-zA-Z]{1,4}[0-9]{1,3}"     # alphanumeric like s3, ec2
    r")$"
)

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "has", "was",
    "were", "are", "our", "your", "their", "its", "not", "but", "you", "we",
    "they", "he", "she", "will", "can", "could", "would", "should",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_vocabulary_candidates(transcript: str) -> List[Dict]:
    """Return candidate DevOps terms found in `transcript` that aren't known yet.

    Pure function — no file I/O. Best-effort heuristics, not guaranteed precision;
    that imprecision is exactly why the human review step in
    scripts/review_vocabulary.py exists rather than auto-applying anything found here.
    """
    if not transcript or not transcript.strip():
        return []

    known_terms, known_terms_set = get_known_terms()
    terms_by_word_count = bucket_terms_by_word_count(known_terms)
    candidates: Dict[str, Dict] = {}

    def _add(phrase: str, signal: str, context_sentence: str, **extra):
        key = phrase.lower().strip()
        if not key or key in known_terms_set or key in candidates:
            return
        candidates[key] = {
            "phrase": phrase.strip(),
            "signal": signal,
            "example_sentence": context_sentence.strip()[:200],
            **extra,
        }

    sentences = re.split(r"(?<=[.!?])\s+", transcript)

    # Signal A: near-miss fuzzy matches against known terms.
    if HAS_TEXTDISTANCE:
        for sentence in sentences:
            words = re.findall(r"\b[\w/]+\b", sentence)
            if not words:
                continue
            max_n = min(3, len(words))
            for n in range(1, max_n + 1):
                for i in range(len(words) - n + 1):
                    ngram = words[i:i + n]
                    # Skip stopwords and very short words individually — these are
                    # exactly what makes jaro-winkler produce noise (see NEAR_MISS_LOW
                    # comment above), not just the joined phrase.
                    if any(w.lower() in STOPWORDS or len(w) < 3 for w in ngram):
                        continue
                    phrase = " ".join(ngram).lower()
                    if len(phrase) < 5 or phrase in known_terms_set:
                        continue
                    best_term, best_score = None, 0.0
                    for target in candidates_for_ngram(terms_by_word_count, n, phrase):
                        score = jaro_winkler_similarity(phrase, target)
                        if score > best_score:
                            best_score, best_term = score, target
                    if best_term and NEAR_MISS_LOW <= best_score < NEAR_MISS_HIGH:
                        _add(
                            " ".join(words[i:i + n]), "near_miss", sentence,
                            near_miss_of=best_term, score=round(float(best_score), 3),
                        )

    # Signal B: novel technical-looking tokens, with extra weight right after a
    # trigger phrase ("we use X", "deployed via X", ...).
    for sentence in sentences:
        for match in TRIGGER_PATTERN.finditer(sentence):
            candidate_word = match.group(2).strip(".,;:")
            if candidate_word.lower() in STOPWORDS or candidate_word.lower() in known_terms_set:
                continue
            _add(candidate_word, "trigger_context", sentence, trigger=match.group(1))

        for token in re.findall(r"\b[\w\-\.]{2,30}\b", sentence):
            clean = token.strip(".,;:")
            lowered = clean.lower()
            if lowered in STOPWORDS or lowered in known_terms_set:
                continue
            if TECHNICAL_TOKEN_PATTERN.match(clean):
                _add(clean, "technical_pattern", sentence)

    return list(candidates.values())


# Candidates are kept per workspace, in the job database (CONTINUUM_DB_PATH,
# so backed up with it and deleted with the workspace). They were one JSON
# file for every customer (CANDIDATES_PATH): one customer's terms sat beside
# another's, stayed after a workspace was deleted, and two worker processes
# writing at once could lose each other's updates (the lock held only within
# a process). That file, if present, is read once into the database as
# candidates from before the split (workspace LEGACY_TENANT): its pending
# ones can still be reviewed, and a phrase rejected there stays rejected for
# every workspace.
LEGACY_TENANT = ""
_SCHEMA = """CREATE TABLE IF NOT EXISTS vocabulary_candidates (
    tenant_id  TEXT NOT NULL,
    phrase_key TEXT NOT NULL,
    record     TEXT NOT NULL,
    PRIMARY KEY (tenant_id, phrase_key)
)"""
_ready = set()
_ready_lock = threading.Lock()


def _db_path() -> str:
    from job_store import DEFAULT_DB_PATH
    return os.getenv("CONTINUUM_DB_PATH", DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    from job_store import ClosingConnection

    path = _db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    db = sqlite3.connect(path, timeout=30, factory=ClosingConnection)
    with _ready_lock:
        if path not in _ready:
            db.execute(_SCHEMA)
            _import_legacy_file(db)
            db.commit()
            _ready.add(path)
    return db


def _import_legacy_file(db: sqlite3.Connection) -> None:
    if not os.path.exists(CANDIDATES_PATH):
        return
    try:
        with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
            legacy = json.load(f)
    except Exception as e:
        logger.warning("Failed to read the old vocabulary candidates file %s: %s", CANDIDATES_PATH, e)
        return
    db.executemany("INSERT OR IGNORE INTO vocabulary_candidates (tenant_id, phrase_key, record) VALUES (?, ?, ?)",
                   [(LEGACY_TENANT, key, json.dumps(entry, ensure_ascii=False))
                    for key, entry in legacy.items() if isinstance(entry, dict)])


def load_candidates(tenant_id: Optional[str] = None) -> List[Dict]:
    """Recorded candidates, each with its "tenant_id": one workspace's, or
    every workspace's (the review script) when `tenant_id` is None."""
    sql, args = "SELECT tenant_id, record FROM vocabulary_candidates", ()
    if tenant_id is not None:
        sql, args = sql + " WHERE tenant_id = ?", (tenant_id,)
    with _connect() as db:
        rows = db.execute(sql + " ORDER BY tenant_id, phrase_key", args).fetchall()
    return [dict(json.loads(record), tenant_id=tenant) for tenant, record in rows]


def record_candidates(candidates: List[Dict], job_id: str, tenant_id: str = "local") -> None:
    """Record detected candidates for the KT's workspace, merging with what
    is already recorded there.

    Never touches glossary.json — only approve_candidate() does that. A
    candidate a human rejected (for this workspace, or before candidates
    were kept per workspace) is not re-added if seen again, so a heuristic
    false-positive doesn't keep nagging the reviewer every run. Terms and
    counts only: no example sentences and no job ids (P0-9); a reviewer sees
    a candidate's context in the KT it came from.
    """
    if not candidates:
        return
    now = _now_iso()
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")          # one writer at a time, across processes
        for cand in candidates:
            key = cand["phrase"].lower()
            rows = dict(db.execute(
                "SELECT tenant_id, record FROM vocabulary_candidates WHERE phrase_key = ? AND tenant_id IN (?, ?)",
                (key, tenant_id, LEGACY_TENANT)).fetchall())
            legacy = json.loads(rows.get(LEGACY_TENANT) or "{}")
            existing = json.loads(rows[tenant_id]) if tenant_id in rows else None
            if (existing or {}).get("status") == "rejected" or legacy.get("status") == "rejected":
                continue
            if existing:
                existing["occurrence_count"] = existing.get("occurrence_count", 1) + 1
                existing["last_seen"] = now
            else:
                existing = {
                    "phrase": cand["phrase"],
                    "signal": cand["signal"],
                    "status": "pending",
                    "occurrence_count": 1,
                    "first_seen": now,
                    "last_seen": now,
                    "near_miss_of": cand.get("near_miss_of"),
                    "score": cand.get("score"),
                    "trigger": cand.get("trigger"),
                }
            db.execute("INSERT OR REPLACE INTO vocabulary_candidates (tenant_id, phrase_key, record) VALUES (?, ?, ?)",
                       (tenant_id, key, json.dumps(existing, ensure_ascii=False)))


def _set_status(phrase: str, tenant_id: str, **fields) -> bool:
    key = phrase.lower()
    with _connect() as db:
        row = db.execute("SELECT record FROM vocabulary_candidates WHERE tenant_id = ? AND phrase_key = ?",
                         (tenant_id, key)).fetchone()
        if row is None:
            logger.warning("No vocabulary candidate %r in workspace %r", phrase, tenant_id)
            return False
        entry = dict(json.loads(row[0]), **fields)
        db.execute("UPDATE vocabulary_candidates SET record = ? WHERE tenant_id = ? AND phrase_key = ?",
                   (json.dumps(entry, ensure_ascii=False), tenant_id, key))
    return True


def approve_candidate(
    phrase: str,
    tenant_id: str,
    target: str = "terms",
    canonical: Optional[str] = None,
    glossary_path: str = "glossary.json",
) -> bool:
    """Promote a workspace's candidate into glossary.json — the live,
    human-approved store that devops_transcription.py's correction pipeline
    reads on every call (for every workspace).

    target: "terms" (recognized vocabulary), "acronyms" (lowercase key -> canonical
    form), or "phrase_corrections" (mishearing -> corrected text; requires `canonical`).
    """
    entry = next((c for c in load_candidates(tenant_id) if c["phrase"].lower() == phrase.lower()), None)
    if entry is None:
        logger.warning("approve_candidate: no such pending candidate: %s", phrase)
        return False

    data = glossary.load_glossary(glossary_path)
    canonical_form = canonical or entry["phrase"]

    if target == "terms":
        if canonical_form not in data.setdefault("terms", []):
            data["terms"].append(canonical_form)
    elif target == "acronyms":
        data.setdefault("acronyms", {})[phrase.lower()] = canonical_form
    elif target == "phrase_corrections":
        if not canonical:
            raise ValueError("approve_candidate: target='phrase_corrections' requires canonical=<corrected text>")
        data.setdefault("phrase_corrections", {})[entry["phrase"]] = canonical
    else:
        raise ValueError(f"approve_candidate: unknown target '{target}'")

    glossary.save_glossary(data, glossary_path)
    return _set_status(phrase, tenant_id, status="approved", approved_as=target, approved_canonical=canonical_form)


def reject_candidate(phrase: str, tenant_id: str) -> bool:
    """Mark a workspace's candidate rejected so it never resurfaces there.
    Does not touch glossary.json."""
    return _set_status(phrase, tenant_id, status="rejected")


def delete_tenant_candidates(tenant_id: str) -> int:
    """Every candidate recorded for a workspace (auth.delete_tenant)."""
    with _connect() as db:
        return db.execute("DELETE FROM vocabulary_candidates WHERE tenant_id = ?", (tenant_id,)).rowcount
