from typing import Any, Dict
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.common import no_coverage_block

COLUMNS = ["Domain", "Coverage", "Assessment"]

COMPLETENESS_INVARIANT = (
    "Every substantive fact must end in exactly one state: mapped, "
    "intentionally deduplicated, or explicitly surfaced as unmapped with a "
    "reason. Silent loss is unacceptable."
)

KNOWLEDGE_GAPS_TITLE = (
    "Knowledge gaps — the following areas were not covered in this KT "
    "session and should be followed up with the outgoing owner"
)


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "KT Coverage & Knowledge Gaps")
    rows = section.get("_coverage_rows") or []
    if not rows:
        return {"section_id": section.get("id"), "section_title": title, "blocks": [no_coverage_block(title)]}

    blocks = []

    # Knowledge coverage (how many transcript FACTS were captured) is a
    # distinct metric from the Coverage matrix below it (which measures
    # TEMPLATE FIELD population) — rendered first and clearly labeled so
    # the two are never mistaken for the same number.
    summary = section.get("_knowledge_coverage_summary") or {}
    if summary.get("verified_against_document"):
        # Counts measured against the rendered document itself (see
        # knowledge_builder.verify_document_coverage), not the pre-render
        # knowledge object.
        text = (
            f"{summary.get('facts_identified', 0)} fact-bearing sentence(s) from the transcript were "
            f"checked against this document: {summary.get('mapped', 0)} appear in a section and "
            f"{summary.get('unmapped', 0)} are listed under Additional Notes."
        )
        if summary.get("recovered"):
            text += (
                f" {summary['recovered']} did not appear anywhere and were added to Additional Notes "
                f"by the final completeness check."
            )
        blocks.append(build_narrative_block("Knowledge coverage", [text]))
    elif summary:
        blocks.append(build_narrative_block("Knowledge coverage", [
            f"{summary.get('facts_identified', 0)} fact-bearing sentence(s) identified in this KT session: "
            f"{summary.get('mapped', 0)} mapped to a section, "
            f"{summary.get('deduplicated', 0)} deduplicated (already captured elsewhere), "
            f"{summary.get('unmapped', 0)} surfaced as unmapped findings, "
            f"{summary.get('lost', 0)} lost."
        ]))

    blocks.append(build_decision_table("Coverage matrix", COLUMNS, rows))
    blocks.append(build_narrative_block("Completeness invariant", [COMPLETENESS_INVARIANT]))

    # Kept visually/structurally separate from open_responsibilities' Open
    # Tasks table: a knowledge gap is "the KT session never covered this,"
    # not "someone agreed to do this" — conflating the two would misrepresent
    # an absence of information as an assigned responsibility.
    gaps = section.get("_knowledge_gaps") or []
    if gaps:
        blocks.append(build_checklist_block(KNOWLEDGE_GAPS_TITLE, gaps))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
