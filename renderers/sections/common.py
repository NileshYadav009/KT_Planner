from typing import Dict, Any, List, Optional
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.warning import build_block as build_warning_block


def _get_structured(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    structured = section.get("_structured")
    return structured if isinstance(structured, dict) else None


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
    structured = _get_structured(section)
    if structured is not None:
        stack = structured.get("monitoring_stack") or []
        first_response = structured.get("first_response_steps") or []
        alert_routing = structured.get("alert_routing")

        tools = []
        for item in stack:
            if not isinstance(item, dict):
                continue
            tool = item.get("tool") or item.get("name") or ""
            monitors = item.get("monitors") or item.get("monitors") or ""
            if tool or monitors:
                tools.append({"label": tool, "value": monitors})

        paragraphs = []
        if alert_routing:
            paragraphs.append(f"Alert routing: {alert_routing}")
        if first_response:
            paragraphs.append("First response steps:")
            paragraphs.extend([f"- {step}" for step in first_response if isinstance(step, str) and step.strip()])

        blocks = []
        if tools:
            blocks.append(build_technology_grid("Monitoring Stack", tools))
        if paragraphs:
            blocks.append(build_narrative_block("Monitoring Notes", paragraphs))
        if blocks:
            return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}

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
    structured = _get_structured(section)
    if structured is not None:
        zones = structured.get("danger_zones") or []
        warnings = []
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            item = zone.get("item") or ""
            why = zone.get("why_dangerous") or ""
            approval = zone.get("approval_required") or ""
            text = item
            if why:
                text += f" — {why}"
            if approval:
                text += f" (Approval required: {approval})"
            if text.strip():
                warnings.append(text)
        if warnings:
            return {"section_id": section.get("id"), "section_title": title, "blocks": [build_warning_block("Danger zones", warnings)]}

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
