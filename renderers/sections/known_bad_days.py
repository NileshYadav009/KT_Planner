from typing import Dict, Any, List
from renderers.blocks.warning import build_block as build_warning_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.common import no_coverage_block


def _coverage_warnings(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Operational Calendar")
    warnings = _coverage_warnings(section)
    cost_patterns = section.get("_cost_patterns") or []

    blocks = []
    if warnings:
        blocks.append(build_warning_block("Known high-risk periods", warnings))
    if cost_patterns:
        blocks.append(build_technology_grid("Cost-related operating patterns", cost_patterns))
    if not blocks:
        blocks.append(no_coverage_block(title))
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
