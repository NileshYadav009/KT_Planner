from typing import Any, Dict
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.common import no_coverage_block

COLUMNS = ["Situation", "Immediate reference"]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Quick Reference")
    rows = section.get("_quick_reference_rows") or []
    if not rows:
        return {"section_id": section.get("id"), "section_title": title, "blocks": [no_coverage_block(title)]}
    return {
        "section_id": section.get("id"),
        "section_title": title,
        "blocks": [build_decision_table("Use this page during an incident or urgent production activity", COLUMNS, rows)],
    }
