from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.ownership import build_block as build_ownership_table
from renderers.blocks.common import no_coverage_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


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

    # Kept as its own paragraph, deliberately never folded into the
    # ownership table above: a general "contact X when unsure" instruction
    # is operational guidance about what to do, not a statement of who
    # formally owns the system (or the on-call tool's name) — conflating
    # them was a real bug (a real transcript's "If you are unsure about a
    # change, involve the appropriate platform or application owner."
    # ended up mislabeled as the On-call tool's NAME in a live PDF).
    guidance_paragraphs: List[str] = []
    if fields.get("operational_escalation_guidance", {}).get("value"):
        guidance_paragraphs.append(fields["operational_escalation_guidance"]["value"])

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if rows:
        blocks.append(build_ownership_table(title, rows))
    if guidance_paragraphs:
        blocks.append(build_narrative_block("Operational escalation guidance", guidance_paragraphs))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
