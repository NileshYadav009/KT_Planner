from typing import Dict, Any, List
from renderers.blocks.warning import build_block as build_warning_block
from renderers.blocks.common import no_coverage_block

# render_monitoring/render_failures/render_generic were removed from here —
# they were dead code (RENDERER_REGISTRY in renderers/sections/__init__.py
# never actually pointed at them; monitoring_observability/common_failures/
# signoff/plain_english_notes all route to dedicated renderer files or were
# removed). render_danger_zones is the one function from this module that's
# still live in the registry.


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render_danger_zones(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Danger Zones")
    warnings = _coverage_paragraphs(section)
    if not warnings:
        return {"section_id": section.get("id"), "section_title": title, "blocks": [no_coverage_block(title)]}
    return {"section_id": section.get("id"), "section_title": title, "blocks": [build_warning_block(title, warnings)]}
