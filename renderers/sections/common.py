from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.warning import build_block as build_warning_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render_monitoring(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Monitoring & Observability")
    coverage_paragraphs = _coverage_paragraphs(section)
    tools: List[Dict[str, str]] = []

    for item in coverage_paragraphs:
        text = str(item)
        if any(tool in text for tool in ["Prometheus", "Grafana", "CloudWatch", "PagerDuty", "ELK", "Datadog"]):
            tools.append({"label": "Tool", "value": text})

    paragraphs = [text for text in coverage_paragraphs if text not in [row["value"] for row in tools]]
    blocks = []
    if tools:
        blocks.append(build_technology_grid("Monitoring Stack", tools))
    if paragraphs:
        blocks.append(build_narrative_block("Monitoring Notes", paragraphs))
    if not blocks:
        blocks.append(build_narrative_block(title, ["No monitoring details captured."]))
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}


def render_failures(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Common Failures & Fixes")
    paragraphs = _coverage_paragraphs(section)

    if not paragraphs:
        paragraphs = ["No failure or incident information captured."]
    blocks = [build_narrative_block(title, paragraphs)]
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}


def render_danger_zones(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Danger Zones")
    warnings = _coverage_paragraphs(section)
    if not warnings:
        warnings = ["No danger zone guidance captured."]
    return {"section_id": section.get("id"), "section_title": title, "blocks": [build_warning_block("Danger zones", warnings)]}


def render_generic(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", section.get("id", "Section"))
    paragraphs = _coverage_paragraphs(section)
    if not paragraphs:
        paragraphs = [f"No content captured for {title}."]
    return {"section_id": section.get("id"), "section_title": title, "blocks": [build_narrative_block(title, paragraphs)]}
