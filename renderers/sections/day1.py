import re
from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.table import build_block as build_decision_table, parse_table_rows
from renderers.blocks.common import no_coverage_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


REQUIRED_ACCESS_COLUMNS = ["Item", "Required", "Location/Link", "Safe on Day-1", "Notes"]
DERIVED_ACCESS_TITLE = "Access to request (derived from the tools named; the KT listed none)"

# An instruction to the new owner, said as one sentence over several things:
# "Please review the deployment pipeline, monitoring dashboards, and
# unresolved recovery documentation before taking ownership." One checklist
# item per thing, the verb and the speaker's words kept (the client's review,
# 13 Nov: the field populator listed the three as "Required access & tools",
# the last with "before taking ownership" in it).
_INSTRUCTION_RE = re.compile(
    r"^\W*(?:(?:so|and|also|then|now|first)[\s,]+)*(?:please\s+)?(?:(?:make\s+sure\s+(?:you|to)|be\s+sure\s+to|you\s+"
    r"(?:should|need\s+to|must|will\s+need\s+to|have\s+to)|i'?d\s+like\s+you\s+to|try\s+to)\s+)?"
    r"(?P<verb>review|read|check|get|request|set\s+up|shadow|learn|go\s+(?:through|over)|walk\s+through|look\s+at|"
    r"confirm|verify|familiari[sz]e\s+yourself\s+with|join|install|study|explore|understand)\s+"
    r"(?P<objects>.+?)(?P<tail>\s+(?:before|after|during|within|by\s+the\s+end\s+of|in\s+(?:your|the)\s+first|"
    r"on\s+(?:your|the)\s+first)\s+[^,]+?)?\s*[.!]?\s*$", re.IGNORECASE)
_OBJECT_SPLIT_RE = re.compile(r",\s*(?:and\s+|or\s+)?|\s+and\s+(?=(?:the\s+|your\s+|our\s+|any\s+)?\w)", re.IGNORECASE)


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def instruction_items(sentence: str):
    """(items, when) for an instruction over a list of things, else
    ([sentence], ""): "Review the deployment pipeline", "Review monitoring
    dashboards", ..., and "before taking ownership"."""
    text = sentence.strip()
    match = _INSTRUCTION_RE.match(text)
    if not match:
        return [text], ""
    objects = [o.strip() for o in _OBJECT_SPLIT_RE.split(match.group("objects")) if o and o.strip()]
    if len(objects) < 2 or any(len(o.split()) > 6 for o in objects):
        return [text], ""
    verb = match.group("verb").strip()
    verb = verb[:1].upper() + verb[1:].lower()
    return [f"{verb} {o}" for o in objects], (match.group("tail") or "").strip()


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Day 1 Survival Checklist")
    fields = section.get("fields", {})

    checklist: List[str] = []
    paragraphs: List[str] = []
    access_rows: List[Dict[str, str]] = []

    stated_blocks: List[Dict[str, Any]] = []
    for sentence in _coverage_paragraphs(section):
        items, when = instruction_items(sentence)
        if len(items) > 1:
            heading = (when[:1].upper() + when[1:]) if when else "First actions stated in the KT"
            # The sentence as said, then one item per thing: every fact is
            # shown in the speaker's words, and the checklist is what to do.
            stated_blocks.append(build_narrative_block(heading, [sentence]))
            stated_blocks.append(build_checklist_block(title, items))      # no heading of its own
    if stated_blocks:
        # The stated instructions are this checklist; a field that read the
        # same sentence as a list of access items, or an LLM's "First-day
        # actions" restating it (the client's KT with Groq, 13 Nov), is not
        # shown beside them.
        said_words = {w for b in stated_blocks for p in (b.get("paragraphs") or []) for w in _words(p)}
        fields = {k: v for k, v in fields.items()
                  if not (k == "required_access"
                          or (k == "first_safe_actions" and _words(str((v or {}).get("value") or "")) <= said_words))}
    derived = section.get("_derived_access") or []

    if fields.get("required_access", {}).get("value"):
        access_rows = parse_table_rows(fields["required_access"]["value"], REQUIRED_ACCESS_COLUMNS)
    if fields.get("first_safe_actions", {}).get("value"):
        checklist.extend([line.strip() for line in str(fields["first_safe_actions"]["value"]).split("\n") if line.strip()])
    if fields.get("actions_not_to_perform", {}).get("value"):
        checklist.extend([f"Do NOT: {line.strip()}" for line in str(fields["actions_not_to_perform"]["value"]).split("\n") if line.strip()])

    blocks = list(stated_blocks)
    if derived and not access_rows:
        # What the KT names but did not list as access to request: derived,
        # and said so (the client's review, 13 Nov: a first-day checklist from
        # the whole transcript, never invented).
        blocks.append(build_decision_table(DERIVED_ACCESS_TITLE, ["Get access to", "Tools the KT names"],
                                           [r for r in derived if isinstance(r, dict)]))
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if access_rows:
        # Natural speech ("review Grafana dashboards, Azure DevOps
        # repositories, AKS namespaces and deployment configurations") gives
        # the ITEMS and nothing else, so parse_table_rows() fills only the
        # first column. Rendered as a table, every remaining column then
        # shows "Not covered during KT" -- and since the table keeps a
        # minimum of two columns, a real generated PDF showed
        # "Grafana dashboards | Not covered during KT", which flatly
        # contradicts itself: the item is listed precisely BECAUSE the
        # transcript named it as required. A list of items is a checklist,
        # not a table; render it as one rather than manufacturing columns
        # the transcript never spoke to.
        item_column = REQUIRED_ACCESS_COLUMNS[0]
        only_items = all(
            not str(row.get(column) or "").strip()
            for row in access_rows
            for column in REQUIRED_ACCESS_COLUMNS[1:]
        )
        if only_items:
            items = [str(row.get(item_column) or "").strip() for row in access_rows]
            items = [item for item in items if item]
            if items:
                blocks.append(build_checklist_block("Required access & tools", items))
        else:
            blocks.append(build_decision_table("Required access & tools", REQUIRED_ACCESS_COLUMNS, access_rows))
    if checklist:
        blocks.append(build_checklist_block("First-day actions", checklist))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
