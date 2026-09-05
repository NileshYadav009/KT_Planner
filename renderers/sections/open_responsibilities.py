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


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    # Informational transition-plan content, not a hazard — unlike
    # danger_zones this section previously always rendered as a red
    # WarningBlock regardless of content, which was misleading.
    title = section.get("title", "Open Responsibilities")
    fields = section.get("fields", {})

    blocks = []
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
