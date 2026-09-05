from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.ownership import build_block as build_ownership_table
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
            rows.append({"role": parts[0], "team": parts[1]})
        else:
            rows.append({"role": text, "team": ""})
    return rows


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "First 30-Day Ownership")
    rows = _coverage_rows(section)
    blocks = []

    if rows:
        blocks.append(build_ownership_table("Ownership handover", rows))
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
