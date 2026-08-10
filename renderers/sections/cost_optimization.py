from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid


def _coverage_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    rows = []
    for item in content:
        text = str(item).strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if len(parts) >= 2:
            rows.append({"label": parts[0], "value": parts[1]})
        else:
            rows.append({"label": text, "value": ""})
    return rows


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Cost Optimization")
    structured = section.get("_structured") or {}
    rows = []
    if structured:
        levers = structured.get("levers") or []
        for lv in levers:
            if not isinstance(lv, dict):
                continue
            name = lv.get("name") or lv.get("lever") or ""
            impact = lv.get("impact") or lv.get("expected_savings") or ""
            notes = lv.get("notes") or ""
            if name:
                rows.append({"label": name, "value": f"Impact: {impact}; {notes}".strip()})
    if not rows:
        rows = _coverage_rows(section)
    blocks = []

    if rows:
        blocks.append(build_technology_grid("Cost optimization levers", rows))
    else:
        content = section.get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        blocks.append(build_narrative_block(title, [str(item).strip() for item in content if str(item).strip()]))

    if not blocks:
        blocks.append(build_narrative_block(title, ["Cost optimization recommendations are being extracted from KT coverage."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
