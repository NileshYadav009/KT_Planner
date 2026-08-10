from typing import Dict, Any, List, Optional
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid


def _get_structured(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    structured = section.get("_structured")
    return structured if isinstance(structured, dict) else None


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Monitoring & Observability")
    fields = section.get("fields", {})

    structured = _get_structured(section)
    paragraphs: List[str] = []
    tech_rows: List[Dict[str, str]] = []

    if structured is not None:
        stack = structured.get("monitoring_stack") or []
        first_response = structured.get("first_response_steps") or []
        alert_routing = structured.get("alert_routing")

        for item in stack:
            if not isinstance(item, dict):
                continue
            tool = item.get("tool") or item.get("name") or ""
            monitors = item.get("monitors") if item.get("monitors") is not None else item.get("monitored_metrics") or ""
            # Normalize monitors to a string
            if isinstance(monitors, list):
                monitors = ", ".join(str(m) for m in monitors if m)
            monitors = str(monitors or "")
            if tool or monitors:
                tech_rows.append({"label": tool, "value": monitors})

        if alert_routing:
            paragraphs.append(f"Alert routing: {alert_routing}")
        if first_response:
            paragraphs.append("First response steps:")
            paragraphs.extend([f"- {step}" for step in first_response if isinstance(step, str) and step.strip()])

    if not tech_rows and fields.get("alerting_tools", {}).get("value"):
        tech_rows.append({"label": "Alerting tools", "value": fields["alerting_tools"]["value"]})
    if not paragraphs and fields.get("monitoring_observability", {}).get("value"):
        paragraphs.append(fields["monitoring_observability"]["value"])

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
