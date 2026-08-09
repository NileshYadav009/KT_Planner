from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.warning import build_block as build_warning_block
from renderers.blocks.technology_grid import build_block as build_technology_grid


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Security Controls")
    fields = section.get("fields", {})

    paragraphs: List[str] = []
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
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(build_narrative_block(title, ["Security controls are being extracted from coverage."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
