from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.common import no_coverage_block

BOOLEAN_CHECK_FIELDS = [
    ("can_deploy", "Replacement can deploy safely"),
    ("understands_rollback", "Understands rollback"),
    ("knows_danger_zones", "Knows danger zones"),
    ("escalation_clear", "Escalation paths are clear"),
    ("architecture_verified", "Architecture verified by new owner"),
]


def _coverage_items(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Handover Completion")
    fields = section.get("fields", {})

    checklist: List[str] = []
    for field_id, label in BOOLEAN_CHECK_FIELDS:
        value = fields.get(field_id, {}).get("value")
        if value is None or value == "":
            continue
        checklist.append(f"{label}: {'Confirmed' if value else 'Not confirmed'}")

    paragraphs: List[str] = []
    kt_status = fields.get("kt_status", {}).get("value")
    if kt_status:
        paragraphs.append(f"KT status: {kt_status}")

    blocks = []
    if checklist:
        blocks.append(build_checklist_block("Handover completion checklist", checklist))
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))

    if not blocks:
        items = _coverage_items(section)
        if items:
            blocks.append(build_checklist_block("Handover completion tasks", items))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
