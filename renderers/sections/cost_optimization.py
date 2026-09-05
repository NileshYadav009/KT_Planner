from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.common import no_coverage_block


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
        # Plain (non-pipe-delimited) text isn't a usable label/value pair — skip
        # it rather than synthesizing a {"value": ""} row, which
        # technology_grid.build_block() silently drops anyway (it requires both
        # label AND value), leaving an empty-but-"meaningful" TechnologyGrid
        # instead of the intended narrative fallback below.
    return rows


def _structured_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    levers = (section.get("_structured") or {}).get("levers")
    if not isinstance(levers, list):
        return []
    rows = []
    for item in levers:
        if not isinstance(item, dict):
            continue
        lever = str(item.get("lever") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if lever:
            rows.append({"label": lever, "value": detail})
    return rows


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Cost Optimization")
    rows = _structured_rows(section) or _coverage_rows(section)
    blocks = []

    if rows:
        blocks.append(build_technology_grid("Cost optimization levers", rows))
    else:
        content = section.get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        paragraphs = [str(item).strip() for item in content if str(item).strip()]
        if paragraphs:
            blocks.append(build_narrative_block(title, paragraphs))

    if not blocks:
        blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
