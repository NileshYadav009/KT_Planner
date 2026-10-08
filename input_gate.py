"""Input quality gate (P0-7): decide whether a transcript can produce a KT
document before any classification runs.

Without it, an empty recording, a few pleasantries or a stand-up uploaded by
mistake all produced an 8-page "KT document", and caption-style text with no
punctuation collapsed into a handful of 60-word chunks that the completeness
check then reported as fully mapped.

Verdicts:
  reject - not enough operational content to build a document from.
  warn   - a document can be built, but the reader is told why it may be thin
           or misplaced (the warning reaches the PDF cover via run_kt_pipeline).
  ok     - nothing to report.
"""
import re
from typing import Dict, List, Optional

MIN_WORDS = 30
MIN_FACT_SENTENCES = 3
THIN_FACT_SENTENCES = 8
# Whisper and meeting-tool exports punctuate; caption streams often don't.
# Above this many words per sentence terminator the text is treated as
# unpunctuated (normal KT speech averages 12-25 words per sentence).
MAX_WORDS_PER_SENTENCE = 45
MIN_WORDS_FOR_PUNCTUATION_CHECK = 60

_TERMINATOR_RE = re.compile(r"[.!?](?:\s|$)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-/.]*")
_DIGIT_RE = re.compile(r"\d")

# Words that mark operational knowledge in a handover. A sentence with one of
# these, a figure, or a named technology counts as fact-bearing.
_OPERATIONAL_WORDS = re.compile(
    r"\b(deploy\w*|release\w*|rollback|roll back|rollout|pipeline\w*|build\w*|merge\w*|"
    r"alert\w*|monitor\w*|dashboard\w*|metric\w*|log\w*|trac\w*|pag\w+|on-?call|"
    r"backup\w*|restore\w*|failover|replica\w*|disaster|recovery|rto|rpo|"
    r"outage\w*|incident\w*|failure\w*|fail\w*|error\w*|latency|throttl\w*|timeout\w*|crash\w*|"
    r"owns?|owner\w*|ownership|escalat\w*|team\w*|"
    r"databases?|cluster\w*|nodes?|pods?|containers?|services?|api\w*|servers?|queues?|cache\w*|"
    r"environments?|staging|production|prod|dev|region\w*|"
    r"secrets?|credentials?|access|permission\w*|certificate\w*|vault|"
    r"runbook\w*|ticket\w*|migration\w*|cost\w*|scal\w+|capacity|traffic|"
    r"never|avoid|must|don't|do not)\b",
    re.IGNORECASE,
)
_PLEASANTRY = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|welcome|good (morning|afternoon|evening)|"
    r"okay|ok|so|right|alright|um+|uh+|yeah|great|cool|any questions|let'?s (get )?start\w*)\b[\s,.!?]*",
    re.IGNORECASE,
)


# Language (P2-7). English prose is about a third function words; a KT in
# another language, or mixed with one ("humara deployment ArgoCD se hota
# hai"), keeps the English product names but few of these.
_ENGLISH_FUNCTION_WORDS = frozenset(
    "the and to of is in it we a an for on that with this are be you our if at from by not have has as or "
    "when then so they there was will can but all it's its into after before which what".split()
)
MIN_ENGLISH_SHARE = 0.12
MIN_WORDS_FOR_LANGUAGE_CHECK = 40
MAX_NON_LATIN_SHARE = 0.3


def language_warning(text: str) -> Optional[str]:
    """A reason to warn when the text does not read as English, else None.
    Continuum maps English KTs; other languages come out wrong or empty."""
    letters = [ch for ch in text or "" if ch.isalpha()]
    non_latin = sum(1 for ch in letters if ord(ch) > 0x24F) / max(1, len(letters))
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    english = sum(1 for w in words if w in _ENGLISH_FUNCTION_WORDS) / max(1, len(words))
    if non_latin > MAX_NON_LATIN_SHARE or (len(words) >= MIN_WORDS_FOR_LANGUAGE_CHECK and english < MIN_ENGLISH_SHARE):
        return ("The transcript does not read as English (or mixes in another language). Continuum maps English "
                "KTs, so sections may be wrong or missing; review every section.")
    return None


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def _is_fact_bearing(sentence: str) -> bool:
    words = _WORD_RE.findall(sentence)
    if len(words) < 4:
        return False
    if _PLEASANTRY.match(sentence) and len(words) < 8 and not _DIGIT_RE.search(sentence):
        return False
    if _DIGIT_RE.search(sentence) or _OPERATIONAL_WORDS.search(sentence):
        return True
    try:
        from field_populator import PATTERN_EXTRACTORS
        tools = PATTERN_EXTRACTORS.get("tools")
        return bool(tools and tools.search(sentence))
    except Exception:
        return False


def assess_transcript(text: str, *, source: str = "paste") -> Dict:
    """Return {"verdict", "reasons", "metrics"} for a cleaned transcript.

    `source` is "paste" (the user can fix the text and resubmit) or "audio"
    (they cannot, so unpunctuated speech is a warning rather than a rejection).
    """
    text = (text or "").strip()
    words = _WORD_RE.findall(text)
    sentences = _sentences(text)
    fact_sentences = [s for s in sentences if _is_fact_bearing(s)]
    terminators = len(_TERMINATOR_RE.findall(text))
    words_per_sentence = len(words) / max(1, terminators)
    metrics = {
        "words": len(words),
        "sentences": len(sentences),
        "fact_bearing_sentences": len(fact_sentences),
        "words_per_sentence": round(words_per_sentence, 1),
    }

    reject: List[str] = []
    warn: List[str] = []
    unpunctuated = (len(words) >= MIN_WORDS_FOR_PUNCTUATION_CHECK
                    and words_per_sentence > MAX_WORDS_PER_SENTENCE)
    if len(words) < MIN_WORDS:
        reject.append(f"The transcript has only {len(words)} words; a KT document needs at least {MIN_WORDS}.")
    elif unpunctuated:
        # Sentence counts are meaningless without sentence boundaries, so the
        # fact-sentence checks below are skipped.
        msg = (f"The transcript has almost no punctuation (about {round(words_per_sentence)} words per sentence), "
               f"so it cannot be split into statements reliably.")
        if source == "paste":
            reject.append(msg + " Paste a punctuated transcript (Teams and Zoom exports are) or upload the recording.")
        else:
            warn.append(msg + " Sections may be misplaced; review every section.")
    elif len(fact_sentences) < MIN_FACT_SENTENCES:
        reject.append(
            f"Only {len(fact_sentences)} sentence(s) carry operational content (systems, deployments, "
            f"alerts, owners, failures); at least {MIN_FACT_SENTENCES} are needed. This does not look like a KT session."
        )
    if not reject and not unpunctuated and MIN_FACT_SENTENCES <= len(fact_sentences) < THIN_FACT_SENTENCES:
        warn.append(
            f"Only {len(fact_sentences)} sentences carry operational content, so most sections will read "
            f"'not covered'. Plan a follow-up session."
        )
    language = language_warning(text)
    if language:
        # With a rejection it explains why; otherwise the reader is warned.
        (reject if reject else warn).insert(0, language)

    verdict = "reject" if reject else ("warn" if warn else "ok")
    return {"verdict": verdict, "reasons": reject or warn, "metrics": metrics}
