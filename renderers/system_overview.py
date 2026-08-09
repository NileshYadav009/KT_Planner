from typing import Dict, Any, List
from renderers.base import build_narrative_block, build_technology_grid


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "System Overview")
    fields = section.get("fields", {})

    paragraphs = []
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
        blocks.append(build_narrative_block(title, ["System overview is being built from extracted knowledge."]))
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
