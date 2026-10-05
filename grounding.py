"""Grounding check for LLM output (P0-4).

Every specific an LLM puts into the KT document (a number, a duration, a
person, a product, a region, a channel, a URL) must appear in the transcript
text it was given. Prompts already say "do not invent", but nothing checked:
in the audit a fake model's invented escalation contact, region, SLA and
tools all reached the PDF unmarked, and malformed replies were printed as
field values.

This is deliberately deterministic and conservative about what it checks:
ordinary words may be paraphrased freely; only "specifics" are verified.
"""
import re
from typing import Iterable, List, Set

try:
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover
    _fuzz = None

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "half": 0.5, "once": 1, "twice": 2,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_DIGITS_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+")
# Identifiers that are never ordinary prose: regions (us-east-1), versions,
# channels, handles, URLs, emails, paths, CLI flags, dotted names.
_IDENTIFIER_RE = re.compile(
    r"https?://\S+|[\w.+-]+@[\w-]+\.[\w.]+|#[\w-]+|@[\w-]+|--[\w-]+|"
    r"\b[a-z]+(?:-[a-z0-9]+)*-\d+[a-z]?\b|\b[\w-]+(?:\.[\w-]+){2,}\b",
    re.IGNORECASE,
)
# Command-line tools are written in lower case, so the capitalisation rule
# does not see them; an invented "kubectl rollout restart" step is as much a
# fabricated specific as an invented product name.
_CLI_TOOLS_RE = re.compile(
    r"\b(kubectl|helm|eksctl|istioctl|gcloud|gsutil|terraform|tofu|pulumi|ansible|docker|podman|"
    r"psql|mongosh|redis-cli|kafka-[a-z-]+|argocd|systemctl|journalctl|curl|ssh|scp|rsync|jq|k9s|"
    r"npm|mvn|gradle)\b"
)
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?:;]\s+|\n\s*|->\s*|\(\s*)$")
# Capitalised words that are ordinary English even mid-sentence in a KT
# field (labels the prompts themselves use, weekdays, months, roles).
_COMMON_CAPS = {
    "i", "a", "an", "the", "and", "or", "not", "high", "medium", "low", "yes", "no", "true", "false",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "step", "week", "day", "production", "staging", "dev",
    "development", "prod", "on-call", "oncall", "api", "ui", "ci", "cd", "sre", "devops", "kt",
    "rto", "rpo", "sla", "slo", "url", "dr", "it", "id", "ok", "eu", "us", "uk", "europe", "asia",
    "america", "north", "south", "east", "west", "complete", "in", "progress", "not_mentioned",
}

_CHATTER_RE = re.compile(
    r"^\s*(sure\b|certainly\b|here(?:'s| is| are)\b|as an ai\b|i (?:can(?:'|no)t|cannot|am unable)\b|"
    r"based on the (?:transcript|fragments|text)\b|the (?:transcript|text) (?:does not|doesn't))",
    re.IGNORECASE,
)
_JSONISH_RE = re.compile(r"[{}\[\]]|\"\s*:")


def _number_values(text: str) -> Set[float]:
    """Numbers written as digits or words: "95", "99.99", "1,500",
    "ninety five", "two hundred thousand", "third"."""
    text = _LIST_MARKER_RE.sub(" ", text or "")
    found: Set[float] = set()
    for m in _DIGITS_RE.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            found.add(float(raw))
        except ValueError:
            pass
    total = current = 0.0
    in_number = False
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _UNITS:
            current += _UNITS[word]
            in_number = True
        elif word in _SCALES and in_number:
            if word == "hundred":
                current *= 100
            else:
                total += current * _SCALES[word]
                current = 0.0
        elif word == "and" and in_number:
            continue
        else:
            if in_number:
                found.add(total + current)
            total = current = 0.0
            in_number = False
    if in_number:
        found.add(total + current)
    return found


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9#@.\-/ ]+", " ", (text or "").lower())).strip()


def _source_words(norm_source: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9\-]*", norm_source))


def _present(term: str, norm_source: str, source_words: Set[str]) -> bool:
    t = _norm(term)
    if not t:
        return True
    if t in norm_source:
        return True
    # Plural/possessive and small spelling drift ("webhooks" vs "webhook",
    # "Trivy" vs "Trivi"): compare word by word with a high threshold.
    t_words = t.split()
    for w in t_words:
        if w in source_words or w.rstrip("s") in source_words or (w + "s") in source_words:
            continue
        if _fuzz is not None and len(w) >= 5 and any(
                abs(len(w) - len(sw)) <= 2 and _fuzz.ratio(w, sw) >= 88 for sw in source_words):
            continue
        return False
    return True


def _specific_terms(value: str) -> List[str]:
    """The checkable specifics in an LLM value: identifiers, capitalised
    names that are not sentence-initial ordinary words, and technologies
    the component catalog recognises."""
    text = _LIST_MARKER_RE.sub("\n", value or "")
    terms: List[str] = [m.group(0) for m in _IDENTIFIER_RE.finditer(text)]
    for m in _WORD_RE.finditer(text):
        word = m.group(0)
        lower = word.lower()
        if lower in _COMMON_CAPS:
            continue
        is_camel = bool(re.search(r"[a-z][A-Z]", word)) or (word.isupper() and len(word) >= 2)
        has_digit = any(ch.isdigit() for ch in word)
        capitalised = word[0].isupper()
        if not (capitalised or has_digit):
            continue
        sentence_initial = bool(_SENTENCE_START_RE.search(text[:m.start()]))
        if sentence_initial and not (is_camel or has_digit):
            continue
        terms.append(word)
    terms.extend(m.group(0) for m in _CLI_TOOLS_RE.finditer(value or ""))
    try:
        from field_populator import PATTERN_EXTRACTORS
        tools = PATTERN_EXTRACTORS.get("tools")
        if tools is not None:
            terms.extend(t for t in tools.findall(value or "") if isinstance(t, str))
    except Exception:
        pass
    return terms


def ungrounded(value, source: str) -> List[str]:
    """Specifics in `value` (str or list of str) that the source text does
    not support. Empty list means the value is grounded."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(ungrounded(item, source))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(ungrounded(item, source))
        return out
    if not isinstance(value, str):
        return []
    norm_source = _norm(source)
    words = _source_words(norm_source)
    missing: List[str] = []
    source_numbers = _number_values(source)
    for n in _number_values(value):
        if n not in source_numbers:
            missing.append(f"{n:g}")
    seen = set()
    for term in _specific_terms(value):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        if not _present(term, norm_source, words):
            missing.append(term)
    return missing


def looks_malformed(value) -> bool:
    """A reply that is model chatter or a JSON fragment rather than a value."""
    if not isinstance(value, str):
        return False
    return bool(_CHATTER_RE.search(value) or _JSONISH_RE.search(value))


def ground_structured(data, source: str, dropped: List[str] = None):
    """Recursively drop ungrounded or malformed values from an LLM JSON
    result. List items are judged one by one so a single invented step does
    not discard the real ones. Records (dicts inside lists, e.g. a failure
    row) are dropped whole if any of their values is ungrounded."""
    dropped = dropped if dropped is not None else []
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                out[key] = ground_structured(value, source, dropped)
            elif isinstance(value, str) and (looks_malformed(value) or ungrounded(value, source)):
                dropped.append(f"{key}: {value}")
                out[key] = None
            else:
                out[key] = value
        return out
    if isinstance(data, list):
        kept = []
        for item in data:
            if isinstance(item, dict):
                bad = [v for v in item.values() if isinstance(v, str) and (looks_malformed(v) or ungrounded(v, source))]
                if bad:
                    dropped.append(str(item))
                    continue
                kept.append(item)
            elif isinstance(item, str):
                if looks_malformed(item) or ungrounded(item, source):
                    dropped.append(item)
                    continue
                kept.append(item)
            else:
                kept.append(item)
        return kept
    return data


def added_specifics(rewrite: str, sources: Iterable[str]) -> List[str]:
    """Specifics a polish rewrite introduced that none of its source
    fragments contain."""
    return ungrounded(rewrite, "\n".join(s for s in sources if isinstance(s, str)))
