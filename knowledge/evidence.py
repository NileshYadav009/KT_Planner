"""Evidence per fact (P1-2): every fact printed in the document points to the
transcript sentence(s) it came from, and to when it was said.

Runs on the finished document (after duplicate removal and the completeness
check). Each fact-bearing unit of a block (a paragraph, list item, warning,
table row, timeline entry) gets an entry in that block's "sources" list, in
the same order as the units: the ids of its source sentences, or [] when
none was found. The sentences themselves (quote, start, end) are listed
once, in document order, in knowledge_object["sources"]; blocks carry only
ids, so no transcript text is copied into the document structure.

A unit is stated when transcript sentences support it: up to three
sentences together contain most of its content words and every number in
it. A value the LLM inferred (INFERRED_MARKER) is never linked: it is shown
as inferred instead. Section status lines ("not covered") and empty table
cells are not facts and get no sources.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from renderers.blocks.common import NOT_COVERED_MESSAGE, SECTION_STATUS_MESSAGES

INFERRED_TEXT = "(inferred"
# Of a unit's content words, the share its source sentences must contain.
STATED_COVERAGE = 0.6
MAX_SOURCES_PER_UNIT = 3
_QUOTE_CHARS = 280

_STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for with from into onto over under about as is are was were be been
being it its this that these those there here we our us you your they their them he she his her i me my mine
do does did done have has had having can could will would should may might must shall not no yes than too very just
also only all any some each every more most other such own same both few many much what which who whom whose when where
why how up down out off again further once because while during before after above below between through until per via
okay ok um uh like really basically actually right well yeah so going get got gets thing things stuff one
""".split())
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'(])")
# A renderer's own words beside a value: "RTO (Recovery Time Objective)".
_PARENTHETICAL = re.compile(r"\s*\([^()]*\)")
# Cells and lines that say nothing was stated are not facts.
_PLACEHOLDER = re.compile(
    r"^\s*(?:not discussed|not covered(?: during kt| in the kt(?: session)?)?|not stated|none stated|unknown|tbd|n/?a|-+)"
    r"\s*\.?\s*$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def build_evidence(sentence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence = []
    for idx, sentence in enumerate(sentence_entries[:3]):
        evidence.append({
            "sentence_index": idx,
            "text": sentence.get("text", ""),
            "start": sentence.get("start"),
            "end": sentence.get("end"),
            "speaker": sentence.get("speaker"),
            "audio_confidence": sentence.get("audio_confidence"),
        })
    return evidence


def _key(word: str) -> str:
    # Word forms compare by their first six letters ("deploys"/"deployment"),
    # as in pdf_rendering's completeness check.
    return word[:6] if len(word) > 6 else word


def _content_keys(text: str) -> List[str]:
    keys, seen = [], set()
    for word in _WORD.findall(str(text or "").lower().replace("*", " ")):
        if word in _STOPWORDS or (len(word) < 3 and not word.isdigit()):
            continue
        k = _key(word)
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _numbers(text: str) -> set:
    return set(_NUMBER.findall(str(text or "")))


_ALIASES: Optional[Tuple[Dict[str, str], Any]] = None


def _aliases() -> Tuple[Dict[str, str], Any]:
    """Every spelling of a product the document may print differently from
    the speaker (the component catalog, and the names knowledge_builder
    gives acronyms: "GKE" -> "Google Kubernetes Engine"), and a regex
    finding them."""
    global _ALIASES
    if _ALIASES is None:
        names: Dict[str, str] = {}
        try:
            from component_catalog import CATALOG

            for product in CATALOG:
                for spelling in (product.name,) + tuple(product.spellings or ()):
                    names[_spelling_key(spelling)] = product.name
        except Exception:
            pass
        try:
            from .knowledge_builder import _CANONICAL_TERM_ALIASES

            for alias, name in _CANONICAL_TERM_ALIASES.items():
                names[_spelling_key(alias)] = name
                names.setdefault(_spelling_key(name), name)
        except Exception:
            pass
        parts = sorted(names, key=len, reverse=True)
        pattern = re.compile(r"(?<![\w.\-])(" + "|".join(r"[\s\-]+".join(map(re.escape, p.split(" "))) for p in parts)
                             + r")(?![\w\-])", re.IGNORECASE) if parts else None
        _ALIASES = (names, pattern)
    return _ALIASES


def _spelling_key(spelling: str) -> str:
    return re.sub(r"[\s\-]+", " ", spelling.strip().lower())


def _canonical(text: str) -> str:
    """Product names written one way on both sides, so "GKE" in the
    transcript supports "Google Kubernetes Engine" in the document."""
    names, pattern = _aliases()
    text = str(text or "")
    if pattern is None:
        return text
    return pattern.sub(lambda m: names.get(_spelling_key(m.group(0)), m.group(0)), text)


def _is_a_name(text: str) -> bool:
    """A one-word cell that names something (a product, team or tool):
    a known product, CamelCase ("PagerDuty") or an acronym ("GKE")."""
    word = text.strip()
    if _spelling_key(word) in _aliases()[0]:
        return True
    return bool(re.fullmatch(r"[A-Za-z][\w./+-]*", word)) and (
        any(c.isupper() for c in word[1:]) or (word.isupper() and len(word) >= 2))


# --------------------------------------------------------------------------
# Transcript sentences with times
# --------------------------------------------------------------------------

def transcript_sentences(transcript: str, segments: Optional[Sequence[Dict[str, Any]]] = None,
                         offset: float = 0.0) -> List[Dict[str, Any]]:
    """The transcript as sentences, each with the time it starts and ends in
    the recording (`segments` are Whisper's; `offset` is leading silence cut
    before transcription). Without segments (a pasted transcript) the times
    are None."""
    if segments:
        text, spans = "", []
        for seg in segments:
            part = str(seg.get("text") or "").strip()
            if not part:
                continue
            if text:
                text += " "
            spans.append((len(text), len(text) + len(part), seg.get("start"), seg.get("end")))
            text += part
    else:
        text, spans = str(transcript or ""), []

    def time_at(pos: int, edge: int) -> Optional[float]:
        for start, end, t0, t1 in spans:
            if start <= pos < end or (pos == end and edge == 1):
                value = t0 if edge == 0 else t1
                return None if value is None else round(float(value) + offset, 2)
        return None

    sentences, pos = [], 0
    for part in _SENTENCE_SPLIT.split(text):
        begin = text.find(part, pos)
        pos = begin + len(part)
        quote = part.strip()
        if len(_content_keys(quote)) < 1:
            continue
        sentences.append({"quote": quote, "start": time_at(begin, 0), "end": time_at(pos, 1) if spans else None})
    return sentences


# --------------------------------------------------------------------------
# Units of a block
# --------------------------------------------------------------------------

def _row_text(row: Dict[str, Any], keys: Optional[Sequence[str]] = None) -> str:
    values = [row.get(k) for k in keys] if keys else list(row.values())
    return " ".join(str(v) for v in values if isinstance(v, str) and v.strip() and v != NOT_COVERED_MESSAGE)


def block_cells(block: Dict[str, Any]) -> Optional[List[List[str]]]:
    """For table-like blocks, each row's cells (what block_units joins)."""
    kind = block.get("type")
    keys = {"TechnologyGrid": ["value"], "OwnershipTable": ["role", "team"],
            "DeploymentTimeline": ["label", "description"]}.get(kind)
    if kind == "DecisionTable":
        keys = block.get("columns") or None
    elif keys is None:
        return None
    rows = block.get("entries") if kind == "DeploymentTimeline" else block.get("rows")
    return [[str(r.get(k)) for k in (keys or list(r)) if isinstance(r.get(k), str) and r.get(k).strip()]
            for r in rows or [] if isinstance(r, dict)]


def block_units(block: Dict[str, Any]) -> Optional[List[str]]:
    """The fact-bearing units of a block, in order; None for blocks that
    hold no transcript facts (diagrams, screenshots, code)."""
    kind = block.get("type")
    if kind == "NarrativeBlock":
        return [str(p) for p in block.get("paragraphs") or []]
    if kind == "ChecklistBlock":
        return [str(i) for i in block.get("items") or []]
    if kind == "WarningBlock":
        return [str(w) for w in block.get("warnings") or []]
    if kind == "TroubleshootingBlock":
        return [str(s) for s in block.get("steps") or []]
    if kind == "TechnologyGrid":
        # The label is a category ("Monitoring"); the value is what was said.
        return [_row_text(r, ["value"]) for r in block.get("rows") or []]
    if kind == "OwnershipTable":
        return [_row_text(r, ["role", "team"]) for r in block.get("rows") or []]
    if kind == "DeploymentTimeline":
        return [_row_text(e, ["label", "description"]) for e in block.get("entries") or []]
    if kind == "DecisionTable":
        columns = block.get("columns") or None
        return [_row_text(r, columns) for r in block.get("rows") or []]
    return None


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

class _Index:
    def __init__(self, sentences: List[Dict[str, Any]], labels: Sequence[str] = (), system_name: str = ""):
        self.sentences = sentences
        canonical = [_canonical(s["quote"]) for s in sentences]
        self.keys = [set(_content_keys(c)) for c in canonical]
        self.numbers = [_numbers(c) for c in canonical]
        # The document's own headings ("COMMON FAILURES & FIXES"): a table
        # cell that is one of them is a pointer, not something said.
        self.labels = {_spelling_key(label) for label in labels if label}
        # The document's name for the system is its own label too: "GCP
        # Data and Machine Learning sends messages via Pub/Sub" is stated by
        # the Pub/Sub sentence, not by the one that names the platform.
        self.system_name = re.compile(re.escape(system_name), re.IGNORECASE) if len(system_name or "") > 3 else None

    def _strip(self, text: str) -> str:
        stripped = _PARENTHETICAL.sub("", text) or text
        if self.system_name is not None:
            without = self.system_name.sub(" ", stripped)
            if _content_keys(without):
                stripped = without
        return _canonical(stripped)

    def sources_for(self, text: str) -> List[int]:
        """Indexes of the sentences that state `text`, or [] if none do."""
        text = self._strip(text)
        keys = _content_keys(text)
        if not keys:
            return []
        numbers = _numbers(text)
        if len(keys) <= 2:
            # A name or a two-word fact: the first sentence that says all of
            # it (where it is introduced), not the shortest that mentions it.
            for i, sent_keys in enumerate(self.keys):
                if set(keys) <= sent_keys and numbers <= self.numbers[i]:
                    return [i]
            return []
        remaining, chosen = set(keys), []
        while remaining and len(chosen) < MAX_SOURCES_PER_UNIT:
            best, best_score = None, (0, 0.0)
            for i, sent_keys in enumerate(self.keys):
                if i in chosen:
                    continue
                gain = len(remaining & sent_keys)
                # Ties go to the more specific sentence (more of it is about this unit).
                score = (gain, gain / (len(sent_keys) or 1))
                if gain and score > best_score:
                    best, best_score = i, score
            # A second or third sentence must add at least two words.
            if best is None or (chosen and best_score[0] < 2):
                break
            if not chosen and best_score[0] < 2:
                return []
            chosen.append(best)
            remaining -= self.keys[best]
        covered = 1 - len(remaining) / len(keys)
        # A short statement needs every content word; a longer one most of them.
        required = STATED_COVERAGE if len(keys) >= 4 else 1.0
        said_numbers = set().union(*(self.numbers[i] for i in chosen)) if chosen else set()
        if covered < required or not numbers <= said_numbers:
            return []
        return sorted(chosen)

    def sources_for_row(self, cells: List[str]) -> List[int]:
        """A table row joins what was said with the document's own labels
        ("Week 1", "Who to page", "No fix was stated"). When the row as a
        whole is not a statement, it is sourced by its cells that are: a
        cell of two or more content words that a sentence states, or a
        one-word cell that names something (a product, tool, team or a
        number), from the sentence that shares most with the row."""
        cells = [c for c in cells if not _PLACEHOLDER.match(c) and _spelling_key(c) not in self.labels]
        if not cells:
            return []
        found = self.sources_for(" ".join(cells))
        if found:
            return found
        # A cell that is one sentence, as said (a quote column), is the source.
        quoted = set()
        for cell in cells:
            keys = set(_content_keys(self._strip(cell)))
            if len(keys) >= 4:
                quoted.update(i for i, k in enumerate(self.keys) if keys <= k)
        if quoted:
            return sorted(quoted)[:MAX_SOURCES_PER_UNIT]
        row_keys = set(_content_keys(_canonical(" ".join(cells))))
        chosen = set()
        for cell in cells:
            keys = _content_keys(self._strip(cell))
            if len(keys) >= 2:
                chosen.update(self.sources_for(cell))
            elif len(keys) == 1 and (_is_a_name(cell) or _NUMBER.fullmatch(cell.strip())):
                holders = [i for i, k in enumerate(self.keys) if keys[0] in k]
                if holders:
                    chosen.add(max(holders, key=lambda i: (len(self.keys[i] & row_keys), -i)))
        return sorted(chosen)[:MAX_SOURCES_PER_UNIT]


def _is_fact(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped not in SECTION_STATUS_MESSAGES and not _PLACEHOLDER.match(stripped)


def attach_evidence(knowledge_object: Dict[str, Any], sentences: List[Dict[str, Any]],
                    skip_sections: Sequence[str] = ("kt_coverage",)) -> Dict[str, int]:
    """Give every fact-bearing unit of the rendered document its sources (see
    the module docstring). Returns counts: units, stated, inferred, unsourced."""
    labels = [sec.get("section_title") for sec in knowledge_object.get("rendered_sections") or []]
    index = _Index(sentences, labels, str(knowledge_object.get("system_name") or ""))
    numbering: Dict[int, int] = {}          # sentence index -> source id, in document order
    stats = {"units": 0, "stated": 0, "inferred": 0, "unsourced": 0}
    for section in knowledge_object.get("rendered_sections") or []:
        if section.get("section_id") in skip_sections:
            continue
        for block in section.get("blocks") or []:
            units = block_units(block)
            if units is None:
                continue
            cells = block_cells(block)
            refs: List[List[int]] = []
            for position, text in enumerate(units):
                if not _is_fact(text):
                    refs.append([])
                    continue
                stats["units"] += 1
                if INFERRED_TEXT in text:
                    stats["inferred"] += 1
                    refs.append([])
                    continue
                found = index.sources_for_row(cells[position]) if cells else index.sources_for(text)
                stats["stated" if found else "unsourced"] += 1
                refs.append(sorted(numbering.setdefault(i, len(numbering) + 1) for i in found))
            block["sources"] = refs
    knowledge_object["sources"] = [
        dict(id=source_id, quote=_short(sentences[i]["quote"]), start=sentences[i]["start"], end=sentences[i]["end"])
        for i, source_id in sorted(numbering.items(), key=lambda kv: kv[1])
    ]
    knowledge_object["_evidence_stats"] = stats
    return stats


def _short(quote: str) -> str:
    if len(quote) <= _QUOTE_CHARS:
        return quote
    cut = quote[:_QUOTE_CHARS].rsplit(" ", 1)[0]
    return cut + "…"


def format_time(seconds: Optional[float]) -> str:
    """[h:]mm:ss, or "" for an untimed (pasted) transcript."""
    if seconds is None:
        return ""
    total = int(seconds)
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def source_lookup(knowledge_object: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {s["id"]: s for s in knowledge_object.get("sources") or []}


def unit_sources(block: Dict[str, Any], position: int) -> List[int]:
    refs = block.get("sources") or []
    return list(refs[position]) if position < len(refs) else []


def evidence_pairs(knowledge_object: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """(unit text, source quotes) for every unit with sources (tests, reports)."""
    lookup = source_lookup(knowledge_object)
    pairs = []
    for section in knowledge_object.get("rendered_sections") or []:
        for block in section.get("blocks") or []:
            units = block_units(block) or []
            for i, text in enumerate(units):
                ids = unit_sources(block, i)
                if ids:
                    pairs.append((text, [lookup[n]["quote"] for n in ids if n in lookup]))
    return pairs
