"""One home per fact (P1-10).

The audit found the same sentence printed several times: Tribal Knowledge
copied every Danger Zones line, Architecture Details repeated System
Overview's paragraphs word for word, and a transcript that said something
three times printed it three times. This pass runs on the rendered document:

- In the real sections, in document order, the first place a sentence
  appears is its home. A later copy is removed from narrative text (from
  the paragraph, not the whole paragraph). Tables and lists are never cut;
  their sentences only count as already shown.
- Digests point instead of copying: a Tribal Knowledge row whose sentence
  has a home elsewhere shows a short excerpt and where the full statement is.
- Text that quotes on purpose is left alone: the Quick Reference (the
  incident card), the coverage page, warnings, captions and the Connections
  evidence table.

Two sentences are the same fact when their words match exactly, or when
nearly all words match in both directions with the same numbers and the
same negations ("RTO is four hours" and "RTO is two hours" are different
facts; so are "deploy on Fridays" and "never deploy on Fridays").
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple

EXEMPT_SECTIONS = frozenset({"quick_reference", "kt_coverage"})
DIGEST_SECTIONS = frozenset({"tribal_knowledge"})
_QUOTING_BLOCK_TYPES = frozenset({"ImageBlock", "DiagramBlock", "CodeBlock"})
_QUOTING_BLOCK_TITLES = frozenset({"Connections stated in the KT"})
# Conflict warnings quote two statements from their section; Danger Zones'
# own warning block is that section's content, not a quote.
_QUOTING_WARNING_PREFIX = "Possible conflict"
WHERE_COLUMN = "Where it is"

_MIN_WORDS = 5
_NEAR_RATIO = 0.9
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=\S)|\n+")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
    "billion half quarter first second third".split())
_NEGATIONS = frozenset({"no", "not", "never", "don't", "doesn't", "isn't", "aren't", "won't", "can't", "cannot",
                        "without", "avoid", "nobody", "none", "nothing"})
_STOP = frozenset("the a an and or to of in on for is are was were be been it that this with as at by we our you your "
                  "so but if then there they its from into also just which will can has have had do does i".split())
# Placeholder text the renderers print in empty sections: not a fact.
_BOILERPLATE_RE = re.compile(r"not covered (?:in|during) the kt|flag it for follow-?up", re.IGNORECASE)


def _split(text: str) -> List[str]:
    return [s.strip() for s in _SPLIT_RE.split(text or "") if s and s.strip()]


class _Facts:
    """Sentences already shown, and where."""

    def __init__(self):
        self._exact: Dict[str, str] = {}
        self._entries: List[Tuple[Set[str], frozenset, frozenset, str]] = []

    @staticmethod
    def _profile(sentence: str):
        words = _WORD_RE.findall(sentence.lower())
        numbers = frozenset(w for w in words if w.isdigit() or w in _NUMBER_WORDS)
        negations = frozenset(w for w in words if w in _NEGATIONS)
        content = {w for w in words if w not in _STOP}
        return words, numbers, negations, content

    def home_of(self, sentence: str) -> Optional[str]:
        words, numbers, negations, content = self._profile(sentence)
        if len(words) < _MIN_WORDS or _BOILERPLATE_RE.search(sentence):
            return None
        exact = self._exact.get(" ".join(words))
        if exact:
            return exact
        for other, other_numbers, other_negations, home in self._entries:
            if numbers != other_numbers or negations != other_negations or not content or not other:
                continue
            shared = len(content & other)
            if shared / len(content) >= _NEAR_RATIO and shared / len(other) >= _NEAR_RATIO:
                return home
        return None

    def add(self, sentence: str, home: str) -> None:
        words, numbers, negations, content = self._profile(sentence)
        if len(words) < _MIN_WORDS or _BOILERPLATE_RE.search(sentence):
            return
        self._exact.setdefault(" ".join(words), home)
        self._entries.append((content, numbers, negations, home))


def _texts(value: Any) -> List[str]:
    """Human-visible strings of a table or list block."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for k, v in value.items() if k not in ("type", "title", "columns") for t in _texts(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _texts(v)]
    return []


def _without(paragraph: str, sentences: List[str]) -> str:
    """The paragraph with those sentences removed, its other text untouched."""
    for sentence in sentences:
        paragraph = paragraph.replace(sentence, "", 1)
    paragraph = re.sub(r"[ \t]{2,}", " ", paragraph)
    paragraph = re.sub(r"\n\s*\n+", "\n", paragraph)
    return paragraph.strip()


def excerpt(text: str, words: int = 8) -> str:
    """The start of a statement, for a digest row that points to it: always
    cut inside the first sentence, so the pointer never repeats a sentence."""
    sentences = _split(text)
    parts = (sentences[0] if sentences else text).split()
    if len(parts) <= 2:
        return " ".join(parts).rstrip(",;:.") + "…"
    keep = min(words, max(3, len(parts) - 2))
    return " ".join(parts[:keep]).rstrip(",;:.") + "…"


def _quotes_on_purpose(block: Dict[str, Any]) -> bool:
    title = str(block.get("title") or "")
    return (block.get("type") in _QUOTING_BLOCK_TYPES or title in _QUOTING_BLOCK_TITLES
            or (block.get("type") == "WarningBlock" and title.startswith(_QUOTING_WARNING_PREFIX)))


def dedupe_rendered_sections(rendered_sections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Give every fact one home in the rendered document (in place).
    Returns counts: sentences removed and digest rows turned into pointers."""
    facts = _Facts()
    stats = {"removed": 0, "referenced": 0}

    for section in rendered_sections or []:
        sid = section.get("section_id")
        if sid in EXEMPT_SECTIONS or sid in DIGEST_SECTIONS:
            continue
        home = section.get("section_title") or sid or "another section"
        kept_blocks, emptied_by = [], None
        for block in section.get("blocks") or []:
            kind = block.get("type")
            if _quotes_on_purpose(block):
                kept_blocks.append(block)
                continue
            if kind != "NarrativeBlock":
                for text in _texts(block):
                    for sentence in _split(text):
                        if not facts.home_of(sentence):
                            facts.add(sentence, home)
                kept_blocks.append(block)
                continue
            paragraphs = []
            for paragraph in block.get("paragraphs") or []:
                if not isinstance(paragraph, str):
                    paragraphs.append(paragraph)
                    continue
                repeats = []
                for sentence in _split(paragraph):
                    earlier = facts.home_of(sentence)
                    if earlier:
                        repeats.append(sentence)
                        emptied_by = emptied_by or earlier
                    else:
                        facts.add(sentence, home)
                if repeats:
                    stats["removed"] += len(repeats)
                    paragraph = _without(paragraph, repeats)
                if paragraph:
                    paragraphs.append(paragraph)
            if paragraphs:
                block["paragraphs"] = paragraphs
                kept_blocks.append(block)
        if not kept_blocks and emptied_by:
            kept_blocks = [{"type": "NarrativeBlock", "title": section.get("section_title") or "",
                            "paragraphs": [f"Everything said about this was already shown under {emptied_by}."]}]
        section["blocks"] = kept_blocks

    for section in rendered_sections or []:
        if section.get("section_id") not in DIGEST_SECTIONS:
            continue
        for block in section.get("blocks") or []:
            rows = block.get("rows")
            if not isinstance(rows, list) or not rows or "Knowledge" not in (block.get("columns") or []):
                continue
            for row in rows:
                text = str(row.get("Knowledge") or "")
                sentences = _split(text) or [text]
                homes = [facts.home_of(s) for s in sentences]
                if all(homes):
                    row["Knowledge"] = excerpt(text)
                    row[WHERE_COLUMN] = homes[0]
                    stats["referenced"] += 1
                else:
                    row[WHERE_COLUMN] = "Only here"
            if WHERE_COLUMN not in block["columns"]:
                block["columns"] = list(block["columns"]) + [WHERE_COLUMN]
    return stats
