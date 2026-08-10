from typing import Dict, Any, List, Optional
from renderers.blocks.warning import build_block as build_warning_block


def _get_structured(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    structured = section.get("_structured")
    return structured if isinstance(structured, dict) else None


def _coverage_warnings(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Danger Zones")
    structured = _get_structured(section)
    if structured is not None:
        zones = structured.get("danger_zones") or []
        warnings = []
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            item = zone.get("item") or ""
            why = zone.get("why_dangerous") or ""
            approval = zone.get("approval_required") or ""
            text = item
            if why:
                text += f" — {why}"
            if approval:
                text += f" (Approval required: {approval})"
            if text.strip():
                warnings.append(text)
        if warnings:
            return {"section_id": section.get("id"), "section_title": title, "blocks": [build_warning_block(title, warnings)]}

    warnings = _coverage_warnings(section)
    if not warnings:
        warnings = ["Danger zones are being inferred from the KT coverage."]
    return {"section_id": section.get("id"), "section_title": title, "blocks": [build_warning_block(title, warnings)]}
