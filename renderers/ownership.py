from typing import Dict, Any, List
from renderers.base import build_narrative_block, build_ownership_table


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Ownership & Escalation")
    fields = section.get("fields", {})

    rows: List[Dict[str, str]] = []
    paragraphs: List[str] = []

    if fields.get("oncall_tool", {}).get("value"):
        rows.append({"role": "On-call tool", "team": fields["oncall_tool"]["value"]})
    if fields.get("escalation_chain", {}).get("value"):
        paragraphs.append(f"Escalation chain: {fields['escalation_chain']['value']}")
    if fields.get("application_ownership", {}).get("value"):
        rows.append({"role": "Application ownership", "team": fields["application_ownership"]["value"]})
    if fields.get("infrastructure_ownership", {}).get("value"):
        rows.append({"role": "Infrastructure ownership", "team": fields["infrastructure_ownership"]["value"]})

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if rows:
        blocks.append(build_ownership_table(title, rows))
    if not blocks:
        blocks.append(build_narrative_block(title, ["Ownership and escalation details are being inferred from extracted knowledge."]))
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
