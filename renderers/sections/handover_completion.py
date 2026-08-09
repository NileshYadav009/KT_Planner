from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.checklist import build_block as build_checklist_block


def _coverage_items(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Handover Completion")
    items = _coverage_items(section)
    blocks = []

    if items:
        blocks.append(build_checklist_block("Handover completion tasks", items))
    else:
        blocks.append(build_narrative_block(title, ["Handover completion items are being derived from KT coverage."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
