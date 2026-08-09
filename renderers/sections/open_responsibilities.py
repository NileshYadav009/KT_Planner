from typing import Dict, Any, List
from renderers.blocks.warning import build_block as build_warning_block


def _coverage_items(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Open Responsibilities")
    items = _coverage_items(section)
    if not items:
        items = ["Open responsibilities are being inferred from the KT coverage."]
    return {"section_id": section.get("id"), "section_title": title, "blocks": [build_warning_block(title, items)]}
