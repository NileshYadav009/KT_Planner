from typing import Dict, Any, List
from renderers.blocks.warning import build_block as build_warning_block


def _coverage_warnings(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Known Bad Days")
    warnings = _coverage_warnings(section)
    if not warnings:
        warnings = ["Known bad day patterns are being identified from the KT coverage."]
    return {"section_id": section.get("id"), "section_title": title, "blocks": [build_warning_block(title, warnings)]}
