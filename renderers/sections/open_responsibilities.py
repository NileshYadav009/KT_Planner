from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.table import build_block as build_decision_table, parse_table_rows
from renderers.blocks.common import no_coverage_block

OPEN_TASKS_COLUMNS = [
    "Task / Responsibility", "Type", "Current Status", "Business Impact",
    "Knowledge Transfer Done", "Recommendation", "Incoming Owner Decision",
]
RECURRING_RESPONSIBILITIES_COLUMNS = ["Activity", "Frequency", "Trigger", "Owner Before", "Owner After"]


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
        if not task:
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
        if not activity:
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
            if rows:
                blocks.append(build_decision_table("Open tasks", OPEN_TASKS_COLUMNS, rows))
        if fields.get("recurring_responsibilities", {}).get("value"):
            rows = parse_table_rows(fields["recurring_responsibilities"]["value"], RECURRING_RESPONSIBILITIES_COLUMNS)
            if rows:
                blocks.append(build_decision_table("Recurring responsibilities", RECURRING_RESPONSIBILITIES_COLUMNS, rows))

    if not blocks:
        items = _coverage_items(section)
        if items:
            blocks.append(build_narrative_block(title, items))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
