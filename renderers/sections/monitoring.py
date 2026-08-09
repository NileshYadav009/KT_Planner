from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Monitoring & Observability")
    fields = section.get("fields", {})

    paragraphs: List[str] = []
    tech_rows: List[Dict[str, str]] = []

    if fields.get("monitoring_observability", {}).get("value"):
        paragraphs.append(fields["monitoring_observability"]["value"])
    if fields.get("alerting_tools", {}).get("value"):
        tech_rows.append({"label": "Alerting tools", "value": fields["alerting_tools"]["value"]})

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if tech_rows:
        blocks.append(build_technology_grid("Monitoring tools", tech_rows))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(build_narrative_block(title, ["Monitoring and observability is being synthesized from available knowledge."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
