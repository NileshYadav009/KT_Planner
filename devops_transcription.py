"""
DevOps-Optimized Whisper Transcription Module

Provides:
1. DevOps vocabulary correction (maps common transcription errors)
2. Post-processing enhancement
3. Technical term preservation
4. Confidence-based error correction
"""

import re
from typing import Dict, List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import textdistance
    HAS_TEXTDISTANCE = True
except ImportError:
    textdistance = None
    HAS_TEXTDISTANCE = False

import glossary
from devops_vocabulary import DEVOPS_VOCABULARY

# ============================================================================
# DevOps Terminology Corrections Dictionary
# ============================================================================
#
# The static, hand-curated term list lives in devops_vocabulary.py. glossary.py's
# glossary.json is a second, persistent layer on top of it — terms a human has
# approved via vocabulary_learning.py's review flow (see scripts/review_vocabulary.py).
# Both are merged here, read live via `glossary.get_glossary()` — which reloads
# from glossary.json when the file's mtime has moved, not just a one-time import —
# so an approval made by a different process (e.g. scripts/review_vocabulary.py)
# takes effect in an already-running server without a restart.

_STATIC_EXTRA_TERMS = [
    "payment orchestration",
    "payment process",
    "approval required",
    "terraform state",
    "kubernetes",
    "docker",
    "jenkins",
    "aws",
    "gcp",
    "azure",
    "production",
    "staging",
    "slack",
    "teams",
    "pagerduty",
]


_KNOWN_TERMS_CACHE: Optional[Tuple[List[str], Set[str]]] = None
_KNOWN_TERMS_CACHE_GLOSSARY_ID: Optional[int] = None


def get_known_terms() -> Tuple[List[str], Set[str]]:
    """Merge the static vocabulary with the live, human-approved glossary.

    Cached and only rebuilt when glossary.get_glossary() actually returns a
    different (reloaded) glossary object — a term approved via
    scripts/review_vocabulary.py is still picked up immediately (get_glossary()
    itself does the mtime check), but this function no longer rebuilds an
    ~1000+-entry set from scratch on every single call. That rebuild was cheap
    when the vocabulary was ~200 terms; after Phase 4's expansion to ~1000+ it
    measurably slowed clean_transcript() (called once per Whisper segment), so
    it's worth caching.
    """
    global _KNOWN_TERMS_CACHE, _KNOWN_TERMS_CACHE_GLOSSARY_ID

    current_glossary = glossary.get_glossary()
    if _KNOWN_TERMS_CACHE is not None and id(current_glossary) == _KNOWN_TERMS_CACHE_GLOSSARY_ID:
        return _KNOWN_TERMS_CACHE

    merged = set(_STATIC_EXTRA_TERMS)
    merged.update(term.lower() for term in DEVOPS_VOCABULARY.keys())
    merged.update(alias.lower() for aliases in DEVOPS_VOCABULARY.values() for alias in aliases)
    merged.update(term.lower() for term in current_glossary.get("terms", []))
    merged.update(acr.lower() for acr in current_glossary.get("acronyms", {}).keys())
    terms = sorted(merged)

    _KNOWN_TERMS_CACHE = (terms, set(terms))
    _KNOWN_TERMS_CACHE_GLOSSARY_ID = id(current_glossary)
    return _KNOWN_TERMS_CACHE


def bucket_terms_by_word_count(terms: List[str]) -> Dict[int, Dict[str, List[str]]]:
    """Index terms by (word count, first letter of the first word) so n-gram
    fuzzy matching only scans same-length, same-first-letter candidates instead
    of the whole (now 1000+ term) vocabulary per n-gram.

    The word-count bucketing alone wasn't enough once the vocabulary grew ~5x in
    Phase 4's second expansion pass — profiling a 3-sentence test transcript
    showed ~56,000 textdistance calls (~750ms) just from clean_transcript(),
    which runs once per Whisper segment during real transcription. The
    first-letter filter is a standard fuzzy-matching "blocking" technique:
    jaro-winkler itself is prefix-weighted, so a target whose first word starts
    with a different letter than the candidate almost never scores near the
    0.88 auto-correct threshold anyway — cutting the comparison pool ~15-20x
    for a small, acceptable chance of missing a same-length correction whose
    very first letter was also mis-transcribed (rare for multi-word phrases,
    and exactly the kind of case the self-learning candidate review exists to
    catch instead of blind auto-correction).
    """
    buckets: Dict[int, Dict[str, List[str]]] = {}
    for term in terms:
        n = len(term.split())
        first_letter = term[0].lower() if term else ""
        buckets.setdefault(n, {}).setdefault(first_letter, []).append(term)
    return buckets


def candidates_for_ngram(buckets: Dict[int, Dict[str, List[str]]], n: int, phrase: str) -> List[str]:
    """Look up the (word count, first letter)-bucketed candidate terms for a
    given n-gram phrase, as produced by bucket_terms_by_word_count()."""
    first_letter = phrase[0].lower() if phrase else ""
    return buckets.get(n, {}).get(first_letter, [])


MIN_FUZZY_PHRASE_WORDS = 2
# Deliberately high: two unrelated short English words routinely score 0.65-0.75
# against each other by coincidence (e.g. "should" vs "build" = 0.70). The
# intended use of multi-word fuzzy matching is recovering a hyphenated term that
# got split into separate tokens (e.g. "auto scal ing" -> "auto scaling"), where
# each word already closely resembles its target — a genuine match doesn't need
# a low bar here.
MIN_PER_WORD_SIMILARITY = 0.82

# ============================================================================
# Phrase-Level Corrections
# ============================================================================

PHRASE_CORRECTIONS = {
    # Common transcription errors
    r"pay[\s-]?bin\s+orchestration": "payment orchestration",
    r"pay[\s-]?bin\s+process": "payment process",
    r"devons": "dev environments",
    r"dev\s+dev": "dev",
    r"broad\s+staging": "prod staging",
    r"broad\s+staging": "prod staging",
    r"qbritis": "kubernetes",
    r"cubernetes": "kubernetes",
    r"cubernetis": "kubernetes",
    r"terra[\s-]?form?\s+state": "terraform state",
    r"flash\s+sales": "flash sales",
    r"flash\s+sale[\s-]?event": "flash sale event",
    r"soby\s+cautious": "so be cautious",
    r"season\s+sales": "flash sales",
    r"teleform": "terraform",
    r"ter[\s-]?form": "terraform",
    r"csed\s+pipeline": "CI/CD pipeline",
    r"csed\s+pipelines?": "CI/CD pipelines",
    r"csed\s+tool": "CI/CD tool",
    r"desklation\s+path": "escalation path",
    r"trevi\b": "Trivy",
    r"argos\s+cd": "ArgoCD",
    r"cache\s+clear": "cache layers",
    r"intra\s+changes": "infrastructure changes",
    r"infrastructure\s+as\s+cold": "Infrastructure as Code",
    r"processrequiresrestoring": "process requires restoring",
    r"disaster\s+recovery\s+processrequiresrestoring": "disaster recovery process requires restoring",
    r"staging\s+environment\s+variables\s+mirrors": "staging environment closely mirrors",
    r"environment\s+variables\s+mirrors": "environment closely mirrors",
    r"non-production\s+environment\s+variables\s+schedule": "non-production environments and schedule",
    r"load\s+traffic": "low traffic",
    r"roll\s+back": "rollback",
    r"helm\s+roll\s+back": "Helm rollback",
    r"cloud\s+front": "CloudFront",
    r"postgres\s+sql": "PostgreSQL",
    r"fast\s+api": "FastAPI",
    r"front[\s-]?end": "frontend",
    r"hel[\s-]?checks": "health checks",
    r"redowning": "redeploying",
    r"redone": "red zone",
    r"ash-skin": "as-is",
    r"understandable\s+back": "understand back",
    r"asclicion": "escalation",
    r"katie": "KT",
    r"katie\s+planner": "KT Planner",
    r"continue": "",  # Remove standalone "continue"
    r"right[\.,]?\s*$": "",  # Remove trailing "Right."
    r"okay[\.,]?\s*$": "",  # Remove trailing "Okay."
    
    # Number and acronym fixes
    r"k[\s-]?8[\s-]?s": "kubernetes",
    r"a[\s-]?w[\s-]?s": "aws",
    r"\bs[\s-]?r[\s-]?e\b": "sre",
    r"g[\s-]?c[\s-]?p": "gcp",
    
    # Tense and grammar
    r"is\s+done": "goes down",
    r"what\s+breaks": "what breaks",
}

# ============================================================================
# Word-Level Corrections (single word mapping)
# ============================================================================

WORD_CORRECTIONS = {
    "qbritis": "kubernetes",
    "cubernetes": "kubernetes",
    "cubernetis": "kubernetes",
    "teleform": "terraform",
    "devons": "dev environments",
    "redowning": "redeploying",
    "redone": "red zone",
    "seekers": "secrets",
    "asclicion": "escalation",
    "desklation": "escalation",
    "csed": "CI/CD",
    "trevi": "Trivy",
    "argos": "Argo",
    "katie": "KT",
    "pay-bin": "payment",
    "paybin": "payment",
    "soby": "so be",
    "understandable": "understand",
    "asclicion": "escalation",
    "hel": "health",
    "grounding": "grounding",
    "flashsale": "flash sale",
    "flashsales": "flash sales",
}

FILLER_WORDS = [
    "okay",
    "actually",
    "basically",
    "right",
    "you know",
    "um",
    "uh",
    "well",
    "so",
    "like",
]

FILLER_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b[\.,]?", re.IGNORECASE)
REPEATED_WORD_PATTERN = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
REPEATED_PHRASE_PATTERN = re.compile(r"\b((?:\w+\s+){1,12}\w+)\s+\1\b", re.IGNORECASE)


def remove_fillers(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(FILLER_PATTERN, "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def remove_repeated_words(text: str) -> str:
    if not text:
        return text
    previous = None
    cleaned = text
    while cleaned != previous:
        previous = cleaned
        cleaned = REPEATED_WORD_PATTERN.sub(r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def remove_repeated_phrases(text: str) -> str:
    if not text:
        return text
    previous = None
    cleaned = text
    while cleaned != previous:
        previous = cleaned
        cleaned = REPEATED_PHRASE_PATTERN.sub(r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_transcript(text: str) -> str:
    """Normalize transcript before semantic classification.

    Steps:
    1. Normalize whitespace
    2. Apply phrase and word corrections
    3. Remove filler words
    4. Collapse repeated words/phrases
    5. Re-apply DevOps corrections over cleaned text

    Args:
        text: Input transcript
    """
    if not text:
        return text

    normalized = re.sub(r"[\r\n\t]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    normalized, _ = apply_devops_corrections(normalized)
    normalized = remove_fillers(normalized)
    normalized = remove_repeated_words(normalized)
    normalized = remove_repeated_phrases(normalized)
    normalized, _ = apply_devops_corrections(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"(\.\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), normalized)
    normalized = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), normalized)
    return normalized

# ============================================================================
# Context-Based Corrections
# ============================================================================

def apply_devops_corrections(text: str) -> Tuple[str, List[Dict]]:
    """
    Apply DevOps-specific transcription corrections.
    
    Returns:
        Tuple of (corrected_text, list_of_corrections)
    """
    if not text:
        return text, []
    
    corrections_applied = []
    corrected = text

    def _apply_phrase_corrections(source: str) -> str:
        updated = source
        for pattern, replacement in PHRASE_CORRECTIONS.items():
            matches = list(re.finditer(pattern, updated, re.IGNORECASE))
            for match in matches:
                original = match.group(0)
                updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
                corrections_applied.append({
                    "type": "phrase",
                    "original": original,
                    "corrected": replacement,
                    "pattern": pattern
                })

        # Human-approved glossary phrases apply on every transcript, not just the
        # low-confidence repair path in context_mapper.py's ContextRepair. These are
        # plain text entered via the approval CLI, not regex, so match them as
        # literal (escaped) phrases rather than raw patterns.
        for phrase, replacement in glossary.get_glossary().get("phrase_corrections", {}).items():
            pattern = re.escape(phrase)
            matches = list(re.finditer(pattern, updated, re.IGNORECASE))
            for match in matches:
                original = match.group(0)
                updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
                corrections_applied.append({
                    "type": "glossary_phrase",
                    "original": original,
                    "corrected": replacement,
                    "pattern": phrase
                })
        return updated

    # Step 1: Phrase corrections before fuzzy/word normalization
    corrected = _apply_phrase_corrections(corrected)

    # Step 2: Apply fuzzy known-term corrections for noisy transcript phrases
    corrected, fuzzy_corrections = apply_fuzzy_term_corrections(corrected)
    corrections_applied.extend(fuzzy_corrections)

    # Step 3: Apply word-level corrections
    words = corrected.split()
    corrected_words = []
    for word in words:
        clean_word = word.rstrip('.,!?;:')
        punct = word[len(clean_word):]

        if clean_word.lower() in WORD_CORRECTIONS:
            original = clean_word
            corrected_word = WORD_CORRECTIONS[clean_word.lower()]
            corrected_words.append(corrected_word + punct)
            corrections_applied.append({
                "type": "word",
                "original": original,
                "corrected": corrected_word,
                "context": word
            })
        else:
            corrected_words.append(word)

    corrected = " ".join(corrected_words)

    # Step 4: Re-apply phrase corrections so fuzzy/word steps cannot undo them
    corrected = _apply_phrase_corrections(corrected)

    corrected = re.sub(r'\s+', ' ', corrected).strip()
    
    return corrected, corrections_applied


def apply_fuzzy_term_corrections(text: str, threshold: float = 0.88) -> Tuple[str, List[Dict]]:
    """Use fuzzy matching against known DevOps terms to correct transcription noise."""
    if not text or not HAS_TEXTDISTANCE:
        return text, []

    corrections = []
    corrected = text
    words = re.findall(r"\b[\w/]+\b", text)
    if not words:
        return text, []

    known_terms, known_terms_set = get_known_terms()
    terms_by_word_count = bucket_terms_by_word_count(known_terms)

    max_n = min(4, len(words))
    for n in range(max_n, MIN_FUZZY_PHRASE_WORDS - 1, -1):
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i+n]
            phrase = " ".join(ngram_words).lower()
            if phrase in known_terms_set:
                continue
            # Skip n-grams where a word is already an exact known term on its own.
            # Without this, an already-correct word can get swept into a fuzzy
            # match with an unrelated neighbor purely by jaro-winkler's prefix
            # sensitivity — e.g. "RabbitMQ heavily" scoring 0.89 against the
            # known alias "rabbit mq" and replacing both words, silently
            # dropping "heavily". This got materially more likely once the
            # vocabulary grew from ~60 to ~700+ terms (Phase 4) — more targets,
            # more coincidental collisions.
            if any(w.lower() in known_terms_set for w in ngram_words):
                continue
            best_term = None
            best_score = 0.0
            for target in candidates_for_ngram(terms_by_word_count, n, phrase):
                target_words = target.split()
                score = textdistance.jaro_winkler.normalized_similarity(phrase, target)
                if score <= best_score:
                    continue
                # Whole-phrase jaro-winkler alone is fooled by a shared leading
                # word: "code is"/"code should" both score ~0.9 against the known
                # alias "code build" purely because "code " is a long shared
                # prefix, even though "is"/"should" bear no resemblance to
                # "build" — the match would silently replace real words. Require
                # every individual word to at least loosely resemble its
                # counterpart too.
                per_word_ok = all(
                    textdistance.jaro_winkler.normalized_similarity(w.lower(), tw) >= MIN_PER_WORD_SIMILARITY
                    for w, tw in zip(ngram_words, target_words)
                )
                if not per_word_ok:
                    continue
                best_score = score
                best_term = target
            if best_term and best_score >= threshold:
                escaped_phrase = re.escape(" ".join(words[i:i+n]))
                pattern = re.compile(rf"\b{escaped_phrase}\b", re.IGNORECASE)
                corrected, count = pattern.subn(best_term, corrected, count=1)
                if count > 0:
                    corrections.append({
                        "type": "fuzzy",
                        "original": phrase,
                        "corrected": best_term,
                        "score": float(best_score)
                    })
    return corrected, corrections


def enhance_segment_with_context(
    text: str,
    previous_text: Optional[str] = None,
    next_text: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Enhance transcription using surrounding context.
    
    This helps disambiguate similar-sounding terms based on surrounding context.
    """
    corrections = []
    
    # Context-based rules
    context_rules = [
        # If mentions deployment near "kubernetes", suggest K8s for any ambiguous terms
        {
            "trigger": r"kubernetes|k8s|docker|container",
            "corrections": {
                "qbritis": "kubernetes",
                "cubernetis": "kubernetes",
            }
        },
        # If mentions AWS near infrastructure, apply AWS context
        {
            "trigger": r"aws|cloud provider",
            "corrections": {
                "a w s": "aws",
            }
        },
    ]
    
    for rule in context_rules:
        full_context = f"{previous_text or ''} {text} {next_text or ''}"
        if re.search(rule["trigger"], full_context, re.IGNORECASE):
            for original, corrected_term in rule["corrections"].items():
                if re.search(original, text, re.IGNORECASE):
                    text = re.sub(original, corrected_term, text, flags=re.IGNORECASE)
                    corrections.append({
                        "type": "context",
                        "original": original,
                        "corrected": corrected_term,
                        "trigger": rule["trigger"]
                    })
    
    return text, corrections


def correct_transcript(
    segments: List[Dict],
    apply_context: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Correct entire transcript segments with DevOps terminology.
    
    Args:
        segments: List of whisper segments
        apply_context: Whether to use surrounding context for better corrections
    
    Returns:
        Tuple of (corrected_segments, statistics)
    """
    corrected_segments = []
    total_corrections = 0
    corrections_by_type = {"phrase": 0, "word": 0, "context": 0}
    
    for idx, segment in enumerate(segments):
        text = segment.get("text", "")
        
        # Get surrounding context
        prev_text = segments[idx - 1].get("text", "") if idx > 0 else None
        next_text = segments[idx + 1].get("text", "") if idx < len(segments) - 1 else None
        
        # Apply corrections
        corrected_text, corrections = apply_devops_corrections(text)
        
        # Apply context-based enhancements
        if apply_context:
            enhanced_text, context_corrections = enhance_segment_with_context(
                corrected_text, prev_text, next_text
            )
            corrected_text = enhanced_text
            corrections.extend(context_corrections)
            corrections_by_type["context"] += len(context_corrections)
        
        # Count corrections
        for correction in corrections:
            corrections_by_type[correction["type"]] = corrections_by_type.get(correction["type"], 0) + 1
            total_corrections += 1
        
        # Create corrected segment
        corrected_segment = segment.copy()
        corrected_segment["text"] = corrected_text
        corrected_segment["corrections"] = corrections
        corrected_segments.append(corrected_segment)
    
    statistics = {
        "total_corrections": total_corrections,
        "by_type": corrections_by_type,
        "segments_corrected": sum(1 for s in corrected_segments if s.get("corrections"))
    }
    
    return corrected_segments, statistics


def get_model_recommendation(duration_seconds: float) -> str:
    """
    Recommend Whisper model size based on audio duration.
    
    Args:
        duration_seconds: Duration of audio in seconds
    
    Returns:
        Model name (tiny, base, small, medium, large)
    """
    # Trade-off between accuracy and speed
    # tiny: ~1-2 min
    # base: ~2-5 min (good for <= 30 min)  ← Recommended for DevOps
    # small: ~5-20 min
    # medium: ~20-60 min
    # large: ~60+ min
    
    if duration_seconds <= 60:  # ≤ 1 min
        return "base"  # Fast enough, better accuracy
    elif duration_seconds <= 600:  # ≤ 10 min
        return "base"  # Recommended for DevOps KT (usually 10-30 min)
    elif duration_seconds <= 1800:  # ≤ 30 min
        return "base"
    elif duration_seconds <= 3600:  # ≤ 1 hour
        return "small"
    else:
        return "small"  # Don't go beyond small for speed


# ============================================================================
# Confidence Scorer
# ============================================================================

def score_transcription_confidence(
    original_text: str,
    corrected_text: str,
    correction_count: int
) -> float:
    """
    Score confidence in transcription (0.0-1.0).
    
    Higher score = more confident (fewer corrections needed).
    """
    if not original_text:
        return 0.5
    
    original_words = original_text.split()
    corrections_ratio = correction_count / max(len(original_words), 1)
    
    # If > 20% of words were corrected, confidence drops
    confidence = max(0.0, 1.0 - (corrections_ratio * 0.5))
    
    return confidence


if __name__ == "__main__":
    # Test examples
    test_texts = [
        "QBritis for container orchestration and terraform for infrastructure provisioning",
        "We use Kubernetes and Docker with pay-bin orchestration",
        "Devons are well-checked before redowning",
    ]
    
    print("=" * 70)
    print("DevOps Transcription Correction Tests")
    print("=" * 70)
    
    for text in test_texts:
        corrected, corrections = apply_devops_corrections(text)
        print(f"\nOriginal:  {text}")
        print(f"Corrected: {corrected}")
        print(f"Corrections: {corrections}")
