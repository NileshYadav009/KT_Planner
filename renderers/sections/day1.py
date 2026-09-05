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


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Day 1 Survival Checklist")
    fields = section.get("fields", {})

    checklist: List[str] = []
    paragraphs: List[str] = []
    access_rows: List[Dict[str, str]] = []

    if fields.get("required_access", {}).get("value"):
        access_rows = parse_table_rows(fields["required_access"]["value"], REQUIRED_ACCESS_COLUMNS)
    if fields.get("first_safe_actions", {}).get("value"):
        checklist.extend([line.strip() for line in str(fields["first_safe_actions"]["value"]).split("\n") if line.strip()])
    if fields.get("actions_not_to_perform", {}).get("value"):
        checklist.extend([f"Do NOT: {line.strip()}" for line in str(fields["actions_not_to_perform"]["value"]).split("\n") if line.strip()])

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if access_rows:
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
