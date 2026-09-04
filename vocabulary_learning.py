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
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import glossary
from devops_transcription import bucket_terms_by_word_count, candidates_for_ngram, get_known_terms

try:
    import textdistance
    HAS_TEXTDISTANCE = True
except ImportError:
    textdistance = None
    HAS_TEXTDISTANCE = False

logger = logging.getLogger(__name__)

CANDIDATES_PATH = os.path.join(os.path.dirname(__file__), "glossary_candidates.json")

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
                        score = textdistance.jaro_winkler.normalized_similarity(phrase, target)
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


def load_candidates(path: str = CANDIDATES_PATH) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load vocabulary candidates %s: %s", path, e)
        return {}


def _save_candidates(data: Dict, path: str = CANDIDATES_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def record_candidates(candidates: List[Dict], job_id: str, path: str = CANDIDATES_PATH) -> None:
    """Persist detected candidates, merging with anything already recorded.

    Never touches glossary.json — only approve_candidate() does that. A candidate
    already marked "rejected" by a human is not re-added if seen again, so a
    heuristic false-positive doesn't keep nagging the reviewer every run.
    """
    if not candidates:
        return

    store = load_candidates(path)
    now = _now_iso()

    for cand in candidates:
        key = cand["phrase"].lower()
        existing = store.get(key)
        if existing and existing.get("status") == "rejected":
            continue

        if existing:
            existing["occurrence_count"] = existing.get("occurrence_count", 1) + 1
            existing["last_seen"] = now
            if job_id not in existing.get("source_jobs", []):
                existing.setdefault("source_jobs", []).append(job_id)
            examples = existing.setdefault("example_contexts", [])
            if cand["example_sentence"] not in examples and len(examples) < 5:
                examples.append(cand["example_sentence"])
        else:
            store[key] = {
                "phrase": cand["phrase"],
                "signal": cand["signal"],
                "status": "pending",
                "occurrence_count": 1,
                "first_seen": now,
                "last_seen": now,
                "source_jobs": [job_id],
                "example_contexts": [cand["example_sentence"]],
                "near_miss_of": cand.get("near_miss_of"),
                "score": cand.get("score"),
                "trigger": cand.get("trigger"),
            }

    _save_candidates(store, path)


def approve_candidate(
    phrase: str,
    target: str = "terms",
    canonical: Optional[str] = None,
    candidates_path: str = CANDIDATES_PATH,
    glossary_path: str = "glossary.json",
) -> bool:
    """Promote a pending candidate into glossary.json — the live, human-approved
    store that devops_transcription.py's correction pipeline reads on every call.

    target: "terms" (recognized vocabulary), "acronyms" (lowercase key -> canonical
    form), or "phrase_corrections" (mishearing -> corrected text; requires `canonical`).
    """
    store = load_candidates(candidates_path)
    key = phrase.lower()
    entry = store.get(key)
    if entry is None:
        logger.warning("approve_candidate: no such pending candidate: %s", phrase)
        return False

    data = glossary.load_glossary(glossary_path)
    canonical_form = canonical or entry["phrase"]

    if target == "terms":
        if canonical_form not in data.setdefault("terms", []):
            data["terms"].append(canonical_form)
    elif target == "acronyms":
        data.setdefault("acronyms", {})[key] = canonical_form
    elif target == "phrase_corrections":
        if not canonical:
            raise ValueError("approve_candidate: target='phrase_corrections' requires canonical=<corrected text>")
        data.setdefault("phrase_corrections", {})[entry["phrase"]] = canonical
    else:
        raise ValueError(f"approve_candidate: unknown target '{target}'")

    glossary.save_glossary(data, glossary_path)

    entry["status"] = "approved"
    entry["approved_as"] = target
    entry["approved_canonical"] = canonical_form
    _save_candidates(store, candidates_path)
    return True


def reject_candidate(phrase: str, candidates_path: str = CANDIDATES_PATH) -> bool:
    """Mark a candidate rejected so it never resurfaces. Does not touch glossary.json."""
    store = load_candidates(candidates_path)
    key = phrase.lower()
    entry = store.get(key)
    if entry is None:
        logger.warning("reject_candidate: no such pending candidate: %s", phrase)
        return False
    entry["status"] = "rejected"
    _save_candidates(store, candidates_path)
    return True
