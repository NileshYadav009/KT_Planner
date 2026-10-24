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
        return _sounds_like_in_source(term, norm_source)
    return True


def _known_tool(term: str) -> bool:
    """The whole term is a tool the component catalog knows."""
    try:
        from field_populator import PATTERN_EXTRACTORS
        tools = PATTERN_EXTRACTORS.get("tools")
    except ImportError:
        return False
    if tools is None:
        return False
    key = re.sub(r"[^a-z0-9]", "", term.lower())
    return any(re.sub(r"[^a-z0-9]", "", str(found).lower()) == key for found in tools.findall(term))


def _sounds_like_in_source(term: str, norm_source: str) -> bool:
    """A known tool name the source holds as a mishearing: a polish fixing
    "post gray SQL" to "PostgreSQL" or "Argo UD" to "Argo CD" was rejected as
    adding facts, so the misheard name stayed in the document. Only names
    the component catalog knows, compared without spaces over one to three
    source words (PostgreSQL 86, Argo CD 83; Datadog/"dataflow" 67)."""
    if _fuzz is None or not _known_tool(term):
        return False
    target = re.sub(r"[^a-z0-9]", "", term.lower())
    if len(target) < 6:
        return False
    words = norm_source.split()
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            window = "".join(words[i:i + n])
            if abs(len(window) - len(target)) <= 3 and _fuzz.ratio(target, window) >= 80:
                return True
    return False


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


def withdrawn_in(source: str) -> list:
    """The values `source` says and then corrects in the same unit
    (devops_transcription.withdrawn_values), line by line."""
    from devops_transcription import withdrawn_values
    return [p for line in (source or "").splitlines() for p in withdrawn_values(line)]


def states_withdrawn_value(text: str, withdrawn: list) -> bool:
    """`text` states a value its source withdrew, measuring the same thing,
    without the value that replaced it: "Database backups retained for seven
    days" from "... seven days—actually, the new retention policy is 30 days"
    (a user's KT, 23 Oct)."""
    if not withdrawn or not isinstance(text, str):
        return False
    from devops_transcription import _value_kinds
    values = [(_number_values(key) or {key}, kinds) for key, kinds in _value_kinds(text)]
    said = set().union(*(v for v, _ in values)) if values else set()
    for pair in withdrawn:
        old, new = _number_values(pair["old"]) or {pair["old"]}, _number_values(pair["new"]) or {pair["new"]}
        if new & said:
            continue
        if any(v & old and kinds & pair["kinds"] for v, kinds in values):
            return True
    return False


def ground_structured(data, source: str, dropped: List[str] = None, _withdrawn: list = None):
    """Recursively drop ungrounded or malformed values from an LLM JSON
    result. List items are judged one by one so a single invented step does
    not discard the real ones. Records (dicts inside lists, e.g. a failure
    row) are dropped whole if any of their values is ungrounded. A value the
    speaker withdrew, shown without its correction, is dropped too."""
    dropped = dropped if dropped is not None else []
    withdrawn = withdrawn_in(source) if _withdrawn is None else _withdrawn

    def bad(value) -> bool:
        return isinstance(value, str) and (looks_malformed(value) or ungrounded(value, source)
                                           or states_withdrawn_value(value, withdrawn))

    def without_withdrawn_lines(key, value):
        # A value of several lines is listed a line each: "...seven days\nNew
        # retention policy is 30 days" printed "seven days" as its own item.
        if not isinstance(value, str) or "\n" not in value or not withdrawn:
            return value
        lines = value.split("\n")
        kept = [line for line in lines if not states_withdrawn_value(line, withdrawn)]
        dropped.extend(f"{key}: {line}" for line in lines if line not in kept)
        return "\n".join(kept) if any(line.strip() for line in kept) else None

    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            value = without_withdrawn_lines(key, value)
            if isinstance(value, (dict, list)):
                out[key] = ground_structured(value, source, dropped, withdrawn)
            elif value is None:
                out[key] = None
            elif bad(value):
                dropped.append(f"{key}: {value}")
                out[key] = None
            else:
                out[key] = value
        return out
    if isinstance(data, list):
        kept = []
        for item in data:
            if isinstance(item, dict):
                if any(bad(v) for v in item.values()):
                    dropped.append(str(item))
                    continue
                kept.append(item)
            elif isinstance(item, str):
                if bad(item):
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


# A negation turns a statement into its opposite, and the word checks
# (ungrounded, ai._fragments_missing_from) ignore it on purpose: "must not be
# changed" -> "never change" is a fair rewrite. So a rewrite that dropped one
# kept every content word and was accepted: "Do not restart the consumers"
# came back as "Restart the consumers", "Never deploy on Fridays" as "Deploy
# on Fridays", "The RTO has not been documented" as "The RTO has been
# documented". Saying the same thing as a prohibition ("avoid", "prohibited")
# counts as keeping it.
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|nobody|nothing|nowhere|neither|nor|cannot|without|avoid\w*|prohibit\w*|forbid\w*|"
    r"refrain\w*)\b|n't\b|"
    # Speech-to-text often drops the apostrophe: "dont restart it".
    r"\b(?:dont|doesnt|didnt|cant|wont|isnt|arent|wasnt|werent|hasnt|havent|shouldnt|couldnt|wouldnt|mustnt)\b",
    re.IGNORECASE,
)
_CONTENT_RE = re.compile(r"[a-z0-9]+")
_POLARITY_FILLER = frozenset(
    "a an and are as at be been but by for from had has have if in into is it its of on or so that the their then "
    "there these this to was were when which while with you your we our they must do does should would will can "
    "also".split())
# Share of a source sentence's content words a line must restate to be
# taken as its rewrite.
_RESTATES = 0.5


def _content(text: str) -> Set[str]:
    words = (w for w in _CONTENT_RE.findall((text or "").lower().replace("n't", " not")))
    return {w[:5] for w in words if w not in _POLARITY_FILLER and not NEGATION_RE.fullmatch(w)}


def negation_changed(source: str, rewrite: str) -> bool:
    """Whether a one-sentence rewrite gained or lost a negation."""
    return bool(NEGATION_RE.search(source or "")) != bool(NEGATION_RE.search(rewrite or ""))


def negations_flipped(sources: Iterable[str], rewrite: str) -> List[str]:
    """Statements a rewrite of several sentences turned into their opposite:
    a source sentence with a negation whose restating line has none, and a
    line with a negation that only restates sentences without one."""
    lines = [line for line in re.split(r"\n+|(?<=[.!?])\s+", rewrite or "") if line.strip()]
    if not lines:
        return []
    line_words = [_content(line) for line in lines]
    sources = [(s, _content(s), bool(NEGATION_RE.search(s))) for s in sources if isinstance(s, str)]
    sources = [(s, words, negated) for s, words, negated in sources if len(words) >= 2]
    flipped = []
    for source, words, negated in sources:
        if not negated:
            continue
        overlap = [len(words & lw) / len(words) for lw in line_words]
        best = max(range(len(lines)), key=overlap.__getitem__)
        if overlap[best] >= _RESTATES and not NEGATION_RE.search(lines[best]):
            flipped.append(source)
    for line, lw in zip(lines, line_words):
        if not NEGATION_RE.search(line):
            continue
        restated = [negated for _, words, negated in sources if len(words & lw) / len(words) >= _RESTATES]
        if restated and not any(restated):
            flipped.append(line)
    return flipped
