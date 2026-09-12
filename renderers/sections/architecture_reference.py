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

    # A handful of narrow fields (architecture_link, last_updated, ...)
    # typically capture far less than the section's raw transcript content
    # — never let a couple of short field-derived lines silently hide a
    # substantially richer raw fallback. Concretely: a gap-fill that mistook
    # "the diagram is in Confluence" for a value of architecture_link once
    # collapsed a real 5-bullet Architecture Reference section down to a
    # single "Architecture reference: Confluence" line, because that one
    # non-empty paragraph was enough to skip the fallback entirely.
    fallback = _coverage_paragraphs(section)
    field_chars = sum(len(p) for p in paragraphs)
    fallback_chars = sum(len(p) for p in fallback)
    prefer_fallback = fallback_chars > field_chars

    blocks = []
    if tech_rows:
        blocks.append(build_technology_grid("Architecture technologies", tech_rows))
    if paragraphs and not prefer_fallback:
        blocks.append(build_narrative_block(title, paragraphs))
    elif fallback:
        blocks.append(build_narrative_block(title, fallback))

    if not blocks:
        blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
