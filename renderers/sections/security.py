from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.warning import build_block as build_warning_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.common import no_coverage_block, tool_names_in


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

    scan_value = fields.get("security_scan_config", {}).get("value")
    scan_tools = tool_names_in(scan_value)
    if scan_tools:
        tech_rows.append({"label": "Security tooling named", "value": ", ".join(scan_tools)})
    elif isinstance(scan_value, str) and scan_value.strip() and len(scan_value.split()) <= 8:
        tech_rows.append({"label": "Security scanning", "value": scan_value.strip()})
    vault_value = fields.get("vault_configuration", {}).get("value")
    if isinstance(vault_value, str) and vault_value.strip():
        # A short value is a tool/product name; a full sentence is rendered
        # as stated rather than wrapped into "handled by <sentence>."
        if len(vault_value.split()) <= 6:
            paragraphs.append(f"Secret management is handled by {vault_value.strip().rstrip('.')}.")
        else:
            paragraphs.append(vault_value.strip())
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
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
