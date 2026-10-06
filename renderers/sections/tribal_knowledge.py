from typing import Any, Dict
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.common import no_coverage_block

# Classification says why it matters. A "Where it is" column is added when
# rows point to the section holding the full statement (document_dedup.py).
COLUMNS = ["Knowledge", "Classification"]


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
