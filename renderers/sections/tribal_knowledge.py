from typing import Any, Dict
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.common import no_coverage_block

COLUMNS = ["Knowledge", "Value", "Classification"]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Tribal Knowledge")
    rows = section.get("_tribal_rows") or []
    if not rows:
        return {"section_id": section.get("id"), "section_title": title, "blocks": [no_coverage_block(title)]}
    return {
        "section_id": section.get("id"),
        "section_title": title,
        "blocks": [build_decision_table("Non-obvious operational knowledge", COLUMNS, rows)],
    }
