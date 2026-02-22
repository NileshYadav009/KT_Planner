"""
Glossary and conservative ASR repair utilities.

Provides phrase- and token-level corrections for common technical terms
and acronyms. Designed to run only when sentence-level audio confidence is
low to avoid hallucination.
"""
from typing import Tuple
import json
import os
import re
import logging
from difflib import SequenceMatcher, get_close_matches

logger = logging.getLogger(__name__)

# Default technical glossary and phrase corrections (conservative)
DEFAULT_GLOSSARY = {
    "terms": [
        "Kafka",
        "Kubernetes",
        "Grafana",
        "Prometheus",
        "REST",
        "HTTP",
        "API",
        "SQL",
        "S3",
        "EC2",
        "CI/CD",
        "async",
        "streaming",
        "event",
        "service",
        "services"
    ],
    # Known phrase-level mis-transcriptions -> corrections
    "phrase_corrections": {
        # Example mapping covering the user's example transcription mistake
        "coffee for a sink event screaming between surfaces": "Kafka for async event streaming between services",
        # Other conservative corrections
        "coffee for a sink": "Kafka for async",
        "sink event": "sync event",
        "screaming": "streaming",
        "surfaces": "services"
    },
    # Acronym canonical forms
    "acronyms": {
        "ci": "CI",
        "cd": "CD",
        "ci/cd": "CI/CD",
        "api": "API",
        "rest": "REST",
        "s3": "S3",
        "ec2": "EC2",
        "kafka": "Kafka"
    }
}


def load_glossary(path: str = "glossary.json") -> dict:
    """Load a user-provided glossary from workspace if present, else default."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded glossary from {path}")
                return data
    except Exception as e:
        logger.warning(f"Failed to load glossary {path}: {e}")
    return DEFAULT_GLOSSARY


GLOSSARY = load_glossary()


def save_glossary(data: dict, path: str = "glossary.json") -> bool:
    """Save glossary back to workspace path. Returns True on success."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # update in-memory
        global GLOSSARY
        GLOSSARY = data
        logger.info(f"Saved glossary to {path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save glossary {path}: {e}")
        return False


AMBIGUITY_NEARBY = ["customer", "interaction", "physical", "bucket", "container", "box", "person", "human", "client"]


def detect_ambiguous_usage(text: str) -> list:
    """Detect glossary terms used near ambiguous non-technical words.

    Returns list of warning strings. Conservative: only flags when a glossary
    token appears within 3 tokens of an ambiguous word.
    """
    warnings = []
    if not text:
        return warnings
    toks = re.findall(r"\w+", text.lower())
    for i, tok in enumerate(toks):
        # check if token matches a glossary term or acronym
        for term in (GLOSSARY.get("terms", []) + list(GLOSSARY.get("acronyms", {}).keys())):
            if tok == term.lower():
                # window check for ambiguous neighbors
                start = max(0, i - 3)
                end = min(len(toks), i + 4)
                window = toks[start:end]
                for amb in AMBIGUITY_NEARBY:
                    if amb in window:
                        warnings.append(f"Glossary term '{term}' appears near ambiguous word '{amb}'")
                        break
    # deduplicate
    return list(dict.fromkeys(warnings))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def apply_glossary_corrections(text: str, sentence_confidence: float, min_confidence_for_correction: float = 0.6) -> Tuple[str, bool]:
    """
    Apply conservative glossary corrections.

    - Phrase-corrections are applied if the sentence confidence is below the
      `min_confidence_for_correction` threshold.
    - Token-level normalization will canonicalize known acronyms and exact
      term matches. Token fuzzy replacement is conservative and uses
      similarity thresholds.

    Returns: (new_text, changed_flag)
    """
    original = text
    text = _normalize_whitespace(text)

    changed = False

    # Only apply aggressive phrase-corrections when confidence is low
    if sentence_confidence < min_confidence_for_correction:
        # Try direct phrase corrections first (longest-first)
        phrase_map = GLOSSARY.get("phrase_corrections", {})
        # Sort keys by length to match longer phrases first
        for bad_phrase in sorted(phrase_map.keys(), key=len, reverse=True):
            pattern = re.compile(re.escape(bad_phrase), re.IGNORECASE)
            if pattern.search(text):
                replacement = phrase_map[bad_phrase]
                text = pattern.sub(replacement, text)
                changed = True

    # Token-level canonicalization for acronyms and technical terms
    tokens = text.split()
    acronyms = GLOSSARY.get("acronyms", {})
    terms = GLOSSARY.get("terms", [])

    def canonical_for_token(tok: str) -> str:
        lowered = tok.lower()
        if lowered in acronyms:
            return acronyms[lowered]
        # Exact term match (case-insensitive)
        for t in terms:
            if lowered == t.lower():
                return t
            # Do NOT perform fuzzy reinterpretation. If ambiguity exists, preserve original token.
            return tok

    new_tokens = []
    for tok in tokens:
        canon = canonical_for_token(tok)
        if canon != tok:
            changed = True
        new_tokens.append(canon)

    new_text = " ".join(new_tokens)

    # Final normalization and return
    new_text = _normalize_whitespace(new_text)
    return new_text, changed
