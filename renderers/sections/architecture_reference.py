from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.common import no_coverage_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Architecture Reference")
    fields = section.get("fields", {})

    tech_rows: List[Dict[str, str]] = []
    paragraphs: List[str] = []

    if fields.get("architecture_link", {}).get("value"):
        paragraphs.append(f"Architecture reference: {fields['architecture_link']['value']}")
    if fields.get("diagram_link", {}).get("value"):
        paragraphs.append(f"Architecture diagram: {fields['diagram_link']['value']}")
    if fields.get("architecture_summary", {}).get("value"):
        paragraphs.append(fields["architecture_summary"]["value"])
    if fields.get("key_components", {}).get("value"):
        tech_rows.append({"label": "Component", "value": fields["key_components"]["value"]})
    if fields.get("platform_services", {}).get("value"):
        tech_rows.append({"label": "Platform service", "value": fields["platform_services"]["value"]})

    blocks = []
    if tech_rows:
        blocks.append(build_technology_grid("Architecture technologies", tech_rows))
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))

    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
