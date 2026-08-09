from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Architecture Reference")
    fields = section.get("fields", {})

    paragraphs: List[str] = []
    tech_rows: List[Dict[str, str]] = []

    if fields.get("architecture_link", {}).get("value"):
        paragraphs.append(f"Architecture documentation: {fields['architecture_link']['value']}")
    if fields.get("last_updated", {}).get("value"):
        paragraphs.append(f"Last updated: {fields['last_updated']['value']}")
    if fields.get("plain_english_notes", {}).get("value"):
        paragraphs.append(fields["plain_english_notes"]["value"])
    if fields.get("key_technologies", {}).get("value"):
        tech_rows.append({"label": "Key technologies", "value": fields["key_technologies"]["value"]})

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if tech_rows:
        blocks.append(build_technology_grid("Architecture technologies", tech_rows))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(build_narrative_block(title, ["Architecture and reference information is being synthesized from the KT content."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
