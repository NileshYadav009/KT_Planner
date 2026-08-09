from typing import Dict, Any, List
from renderers.base import build_narrative_block, build_warning_block, build_technology_grid


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Security Controls")
    fields = section.get("fields", {})

    paragraphs = []
    warnings: List[str] = []
    tech_rows: List[Dict[str, str]] = []

    if fields.get("security_scan_config", {}).get("value"):
        tech_rows.append({"label": "Security scanner", "value": fields["security_scan_config"]["value"]})

    if fields.get("vault_configuration", {}).get("value"):
        paragraphs.append(f"Secret management is handled by {fields['vault_configuration']['value']}.")

    if fields.get("security_issues", {}).get("value"):
        warnings.append(fields["security_issues"]["value"])

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if tech_rows:
        blocks.append(build_technology_grid("Security tools", tech_rows))
    if warnings:
        blocks.append(build_warning_block("Security warnings", warnings))
    if not blocks:
        blocks.append(build_narrative_block(title, ["Security controls are being extracted from coverage."]))
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
