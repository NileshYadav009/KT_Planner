"""Corrections, superseded tools and contradictions in a KT session (P0-5).

Speakers correct themselves ("The RTO is four hours. Actually the RTO is
thirty minutes, the four hours was the old target.") and describe changes
("Alerts go to Opsgenie now, we moved off PagerDuty last month."). Before
this module the first value always won: the PDF headlined the withdrawn RTO
and named the abandoned paging tool, and "Deploying on Fridays is fine" sat
next to "Never deploy on Fridays" with nothing flagging it.
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

# A statement that corrects or updates what came before it.
CORRECTION_RE = re.compile(
    r"\b(?:actually|correction|sorry|i\s+mean|to\s+be\s+(?:clear|precise)|now|these\s+days|nowadays|"
    r"currently|as\s+of|since\s+the\s+(?:reorg|migration|move)|since\s+last\s+\w+|updated|changed\s+(?:it\s+)?to|"
    r"new\s+target|instead)\b",
    re.IGNORECASE,
)
# A tool someone stopped using: the tool named right after the phrase is the
# old one.
_SUPERSEDE_RE = re.compile(
    r"\b(?:moved\s+(?:off|away\s+from|from)|migrated\s+(?:off\s+|away\s+)?from|switched\s+(?:off\s+|away\s+)?from|"
    r"replaced|stopped\s+using|no\s+longer\s+use[sd]?|used\s+to\s+use|retired|decommissioned|got\s+rid\s+of)\b",
    re.IGNORECASE,
)
_PROHIBIT_RE = re.compile(r"\b(?:never|do\s+not|don'?t|avoid|must\s+not|mustn'?t|not\s+allowed|forbidden|no)\b", re.IGNORECASE)
_PERMIT_RE = re.compile(r"\b(?:is|are|it'?s)\s+(?:fine|ok|okay|allowed|safe)\b|\bwe\s+(?:can|do|usually)\b|\bfeel\s+free\b",
                        re.IGNORECASE)
_STOP = set("the a an is are was were to of for in on at and or but we our it this that be with by from as us you "
            "your fine ok okay never not do don't avoid must allowed can usually no".split())


def _norm_value(value: str) -> str:
    try:
        from grounding import _number_values
        nums = sorted(_number_values(value))
    except Exception:
        nums = []
    unit = re.search(r"\b(second|minute|hour|day|week|month)s?\b", value or "", re.IGNORECASE)
    if nums:
        return f"{nums[0]:g} {unit.group(1).lower() if unit else ''}".strip()
    return re.sub(r"\s+", " ", (value or "").strip().lower())


# A value the speaker explicitly calls outdated. Transcript cleaning strips
# fillers such as "actually", so "the four hours was the old target" is often
# the only surviving signal of a correction.
_UNIT = r"(?:second|minute|hour|day|week|month)s?"
_DURATION = r"(?P<val>(?:(?!" + _UNIT + r"\b)[\w.-]+\s+){1,3}?" + _UNIT + r")"
_OLD_VALUE_RES = [
    re.compile(_DURATION + r"\s+(?:was|were|is)\s+the\s+(?:old|previous|former)\b", re.IGNORECASE),
    re.compile(r"\b(?:used\s+to\s+be|previously|formerly|was\s+originally)\s+" + _DURATION, re.IGNORECASE),
]


def _old_values(sentences: Sequence[str]) -> set:
    old = set()
    for sentence in sentences:
        for rx in _OLD_VALUE_RES:
            for m in rx.finditer(sentence or ""):
                old.add(_norm_value(m.group("val")))
    return old


def resolve_single_value(candidates: Sequence[Tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """Choose one value for a single-valued fact (RTO, RPO, ...) from
    (value, sentence) candidates in transcript order.

    Returns (value, note). One distinct value: no note. Several: a value
    stated with a correction marker ("Actually the RTO is ...") wins, the
    latest such one, and the note names what it replaced. Several with no
    correction marker: the value is reported as a conflict to confirm."""
    seen: Dict[str, Tuple[str, str]] = {}
    for value, sentence in candidates:
        key = _norm_value(value)
        if key:
            seen.setdefault(key, (value, sentence))
    if not seen:
        return None, None
    distinct = list(seen.values())
    if len(distinct) == 1:
        return distinct[0][0], None
    old = _old_values([s for _, s in candidates])
    current = [(v, s) for v, s in distinct if _norm_value(v) not in old]
    if old and len(current) == 1:
        replaced = [v for v, _ in distinct if _norm_value(v) in old]
        return current[0][0], "replaces " + " and ".join(replaced) + ", stated in the session as the old value"
    corrected = [(v, s) for v, s in candidates if CORRECTION_RE.search(s)]
    if corrected:
        chosen = corrected[-1][0]
        older = [v for v, _ in distinct if _norm_value(v) != _norm_value(chosen)]
        return chosen, "replaces " + " and ".join(older) + ", stated earlier in the session"
    values = "; ".join(v for v, _ in distinct)
    return f"Conflicting values stated: {values}", "conflict - confirm with the outgoing owner"


def find_superseded_tools(texts: Sequence[str]) -> Dict[str, Dict[str, Optional[str]]]:
    """{old_tool_lower: {"old", "new", "quote"}} for tools the session says
    were replaced or dropped."""
    try:
        from field_populator import PATTERN_EXTRACTORS
        matcher = PATTERN_EXTRACTORS["tools"]
    except Exception:
        return {}
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for text in texts:
        for m in _SUPERSEDE_RE.finditer(text or ""):
            spans = matcher.spans(text) if hasattr(matcher, "spans") else []
            after = [s for s in spans if s[0] >= m.end()]
            if not after:
                continue
            old = after[0]
            # The replacement: "... with Y" / "... to Y" after the old tool,
            # otherwise a tool named before the phrase ("Alerts go to
            # Opsgenie now, we moved off PagerDuty").
            tail = text[old[1]:]
            to_new = re.match(r"\s*(?:to|with|for)\s+", tail)
            new = None
            if to_new:
                later = [s for s in spans if s[0] >= old[1]]
                new = later[0][2] if later else None
            if new is None:
                before = [s for s in spans if s[1] <= m.start() and s[2].lower() != old[2].lower()]
                new = before[-1][2] if before else None
            out[old[2].lower()] = {"old": old[2], "new": new, "quote": text.strip()}
    return out


def _content_words(text: str) -> set:
    # Compared on a 5-letter stem so "deploying" and "deploy" are the same word.
    return {w.rstrip("s")[:5] for w in re.findall(r"[a-z][a-z']+", (text or "").lower()) if w not in _STOP and len(w) > 2}


def find_contradictions(texts: Sequence[str]) -> List[Tuple[str, str]]:
    """Pairs of sentences that permit and forbid the same thing ("Deploying
    on Fridays is fine" / "Never deploy on Fridays"). Requires two shared
    content words so unrelated rules are never paired."""
    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            pa, fa = bool(_PERMIT_RE.search(a)), bool(_PROHIBIT_RE.search(a))
            pb, fb = bool(_PERMIT_RE.search(b)), bool(_PROHIBIT_RE.search(b))
            if not ((pa and not fa and fb and not pb) or (pb and not fb and fa and not pa)):
                continue
            if len(_content_words(a) & _content_words(b)) >= 2:
                pairs.append((a.strip(), b.strip()))
    return pairs
