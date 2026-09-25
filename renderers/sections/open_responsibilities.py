import re
from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.table import build_block as build_decision_table, parse_table_rows
from renderers.blocks.common import no_coverage_block

OPEN_TASKS_COLUMNS = [
    "Task / Responsibility", "Type", "Current Status", "Business Impact",
    "Knowledge Transfer Done", "Recommendation", "Incoming Owner Decision",
]
RECURRING_RESPONSIBILITIES_COLUMNS = ["Activity", "Frequency", "Trigger", "Owner Before", "Owner After"]

# Conditional/hedge phrasing a real task description essentially never
# opens with ("Finish migrating the batch job to EKS" is imperative/noun-
# phrase shaped) but a generic safety-escalation reminder almost always
# does — confirmed as the single most common raw-table-fallback offender
# across many real generated KTs: "If you are unsure about an ongoing
# production activity, Contact platform engineering before proceeding."
# recurs near-verbatim across unrelated transcripts, is short enough (~13
# words) to slip past a word-count-only guard, and is already correctly
# excluded by name in llm/prompts.py's open_responsibilities prompt ("Do
# NOT include general safety warnings, escalation/contact instructions") —
# this mirrors that same judgment for when the LLM path didn't run at all.
_GUIDANCE_PHRASE_RE = re.compile(
    r"^\s*(if\s+you\s+are\s+unsure|if\s+you're\s+unsure|if\s+unsure|"
    r"regarding\s+open\s+responsibilities)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"\.\s+[A-Z]")


def _is_generic_guidance(text: str) -> bool:
    """True when `text` opens with hedge/conditional phrasing that marks it
    as a generic safety-escalation reminder rather than a task — applied to
    BOTH the raw-fallback path and the LLM's own structured output.

    The open_responsibilities structured prompt (llm/prompts.py) already
    tells the model by name not to include this class of content ("Do NOT
    include general safety warnings, escalation/contact instructions"), but
    that's a soft instruction a smaller/faster model doesn't always follow —
    confirmed live: qwen3-8b (via Groq) put BOTH a real transition-plan
    sentence AND "If you are unsure about an ongoing production activity,
    contact platform engineering before proceeding." into "open_tasks" for
    the same transcript where it correctly excluded that phrase from a
    different section. Applying this filter to the model's own output too
    (not just the no-LLM fallback) is defense-in-depth against exactly that
    class of instruction-following miss, not just a fallback-only patch."""
    return bool(_GUIDANCE_PHRASE_RE.match(text.strip()))


def _looks_like_narrative_not_a_task(text: str) -> bool:
    """True when a raw-fallback row's only populated cell reads as
    unstructured prose/guidance rather than a discrete task or recurring-
    duty label — see the module-level comment on _GUIDANCE_PHRASE_RE and
    the multi-sentence case in render()'s docstring-equivalent comment
    below for the two real bugs this guards against."""
    text = text.strip()
    if not text:
        return False
    if _is_generic_guidance(text):
        return True
    if _SENTENCE_BOUNDARY_RE.search(text):
        # More than one sentence — a genuine task/duty label is virtually
        # always a single clause, even when lifted verbatim from speech.
        return True
    return len(text.split()) > 20


def _coverage_items(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def _structured_task_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    tasks = (section.get("_structured") or {}).get("open_tasks")
    if not isinstance(tasks, list):
        return []
    rows = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task") or "").strip()
        if not task or _is_generic_guidance(task):
            continue
        rows.append({
            "Task / Responsibility": task,
            "Type": str(item.get("type") or "").strip(),
            "Current Status": str(item.get("status") or "").strip(),
            "Business Impact": str(item.get("business_impact") or "").strip(),
            "Knowledge Transfer Done": str(item.get("kt_done") or "").strip(),
            "Recommendation": str(item.get("recommendation") or "").strip(),
            "Incoming Owner Decision": str(item.get("owner_decision") or "").strip(),
        })
    return rows


def _structured_recurring_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    items = (section.get("_structured") or {}).get("recurring_responsibilities")
    if not isinstance(items, list):
        return []
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        activity = str(item.get("activity") or "").strip()
        if not activity or _is_generic_guidance(activity):
            continue
        rows.append({
            "Activity": activity,
            "Frequency": str(item.get("frequency") or "").strip(),
            "Trigger": str(item.get("trigger") or "").strip(),
            "Owner Before": str(item.get("owner_before") or "").strip(),
            "Owner After": str(item.get("owner_after") or "").strip(),
        })
    return rows


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    # Informational transition-plan content, not a hazard — unlike
    # danger_zones this section previously always rendered as a red
    # WarningBlock regardless of content, which was misleading.
    title = section.get("title", "Open Responsibilities")
    fields = section.get("fields", {})

    blocks = []

    # Structured extraction (llm/prompts.py's open_responsibilities prompt)
    # is preferred: it's instructed to only emit rows for genuine tasks/
    # recurring duties and to leave out general safety/escalation guidance
    # that isn't actually a task — unlike the raw type:"table" fallback,
    # which has no way to tell a stray sentence apart from a real task and
    # previously dumped things like production-safety warnings in here with
    # every other column blank.
    structured_task_rows = _structured_task_rows(section)
    structured_recurring_rows = _structured_recurring_rows(section)
    if structured_task_rows:
        blocks.append(build_decision_table("Open tasks", OPEN_TASKS_COLUMNS, structured_task_rows))
    if structured_recurring_rows:
        blocks.append(build_decision_table("Recurring responsibilities", RECURRING_RESPONSIBILITIES_COLUMNS, structured_recurring_rows))

    if not blocks:
        if fields.get("open_tasks", {}).get("value"):
            rows = parse_table_rows(fields["open_tasks"]["value"], OPEN_TASKS_COLUMNS)
            # A single row whose only populated cell is unstructured prose
            # or generic guidance is not a task — it's a whole raw sentence
            # that parse_table_rows (a last-resort, `\n`-split fallback with
            # no structure-awareness of its own) had no way to split into
            # real task rows. Rendering it as a table row makes it *look*
            # like structured data ("Task / Responsibility: <sentence>",
            # every other column blank) when it's really unstructured
            # narrative or safety-escalation advice — confirmed on real KTs
            # both for a long multi-sentence transition-plan paragraph and,
            # far more often, a short-but-generic "If you are unsure about
            # an ongoing production activity, contact platform engineering
            # before proceeding." sentence that a word-count-only check
            # doesn't catch. Fall through to the narrative block below.
            if len(rows) == 1 and _looks_like_narrative_not_a_task(
                str(rows[0].get(OPEN_TASKS_COLUMNS[0], ""))
            ):
                rows = []
            if rows:
                blocks.append(build_decision_table("Open tasks", OPEN_TASKS_COLUMNS, rows))
        if fields.get("recurring_responsibilities", {}).get("value"):
            rows = parse_table_rows(fields["recurring_responsibilities"]["value"], RECURRING_RESPONSIBILITIES_COLUMNS)
            if len(rows) == 1 and _looks_like_narrative_not_a_task(
                str(rows[0].get(RECURRING_RESPONSIBILITIES_COLUMNS[0], ""))
            ):
                rows = []
            if rows:
                blocks.append(build_decision_table("Recurring responsibilities", RECURRING_RESPONSIBILITIES_COLUMNS, rows))

    if not blocks:
        items = _coverage_items(section)
        if items:
            blocks.append(build_narrative_block(title, items))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
