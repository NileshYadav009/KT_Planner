from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.common import no_coverage_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "System Overview")
    fields = section.get("fields", {})

    paragraphs: List[str] = []
    tech_rows: List[Dict[str, str]] = []

    if fields.get("cache_layer", {}).get("value"):
        tech_rows.append({"label": "Cache Layer", "value": fields["cache_layer"]["value"]})
    if fields.get("event_streaming", {}).get("value"):
        tech_rows.append({"label": "Event Streaming", "value": fields["event_streaming"]["value"]})
    if fields.get("business_criticality", {}).get("value"):
        paragraphs.append(f"Business criticality is {fields['business_criticality']['value']}.")
    if fields.get("documentation_links", {}).get("value"):
        paragraphs.append(f"Documentation references: {fields['documentation_links']['value']}")
    if fields.get("system_description", {}).get("value"):
        paragraphs.append(fields["system_description"]["value"])

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if tech_rows:
        blocks.append(build_technology_grid("Key Technology", tech_rows))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
