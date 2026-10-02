from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.common import no_coverage_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Sign-off")
    fields = section.get("fields", {})

    rows: List[Dict[str, str]] = []
    if fields.get("outgoing_owner", {}).get("value"):
        rows.append({"label": "Outgoing owner", "value": str(fields["outgoing_owner"]["value"])})
    if fields.get("incoming_owner", {}).get("value"):
        rows.append({"label": "Incoming owner", "value": str(fields["incoming_owner"]["value"])})
    if fields.get("date", {}).get("value"):
        rows.append({"label": "Sign-off date", "value": str(fields["date"]["value"])})
    approved = fields.get("approved", {}).get("value")
    # Only a real boolean is a statement about approval. An unfilled field
    # (value "") used to render "Not yet" -- a claim the transcript never made.
    if isinstance(approved, bool):
        rows.append({"label": "Approved by incoming owner", "value": "Yes" if approved else "Not yet"})

    blocks = []
    if rows:
        blocks.append(build_technology_grid("Sign-off record", rows))
    else:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
