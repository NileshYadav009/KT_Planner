from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.common import no_coverage_block


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render monitoring section with structured data if available.
    Falls back to fields and coverage_content if structured data is missing.
    """
    title = section.get("title", "Monitoring & Observability")
    blocks = []
    
    # Priority 1: Try structured extraction (tools, first_response_steps, alert_routing)
    structured = section.get("_structured", {})
    if isinstance(structured, dict) and structured:
        # Tools/Monitoring stack
        tools = structured.get("tools", [])
        if tools:
            tech_rows = [{"label": "Tools", "value": ", ".join(tools)}]
            blocks.append(build_technology_grid("Monitoring Stack", tech_rows))
        
        # First response steps
        first_response = structured.get("first_response_steps", [])
        if first_response:
            blocks.append(build_checklist_block("First Response Steps", first_response))
        
        # Alert routing (as narrative)
        alert_routing = structured.get("alert_routing")
        if alert_routing:
            blocks.append(build_narrative_block("Alert Routing", [alert_routing]))
    
    # Priority 2: Try populated fields
    fields = section.get("fields", {})
    if not blocks:
        paragraphs = []
        if fields.get("monitoring_observability", {}).get("value"):
            paragraphs.append(fields["monitoring_observability"]["value"])
        if fields.get("alerting_tools", {}).get("value"):
            tech_rows = [{"label": "Alerting tools", "value": fields["alerting_tools"]["value"]}]
            blocks.append(build_technology_grid("Monitoring tools", tech_rows))
        
        if paragraphs:
            blocks.append(build_narrative_block(title, paragraphs))
    
    # Priority 3: Fall back to coverage_content
    if not blocks:
        coverage_content = section.get("coverage_content") or []
        if isinstance(coverage_content, str):
            coverage_content = [coverage_content]
        fallback_text = [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]
        
        if fallback_text:
            blocks.append(build_narrative_block(title, fallback_text))
        else:
            blocks.append(no_coverage_block(title))
    
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}

