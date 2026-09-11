from typing import Any, Dict, List

from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.common import no_coverage_block


def _coverage_items(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Additional Notes (Unmapped Findings)")
    items = _coverage_items(section)
    if not items:
        return {"section_id": section.get("id"), "section_title": title, "blocks": [no_coverage_block(title)]}
    return {"section_id": section.get("id"), "section_title": title, "blocks": [build_checklist_block(title, items)]}
