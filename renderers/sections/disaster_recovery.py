from typing import Dict, Any, List
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.warning import build_block as build_warning_block
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.common import no_coverage_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Disaster Recovery")
    fields = section.get("fields", {})

    checklist_items: List[str] = []
    warnings: List[str] = []
    paragraphs: List[str] = []

    if fields.get("rto_steps", {}).get("value"):
        checklist_items.extend([x.strip() for x in str(fields["rto_steps"]["value"]).split("\n") if x.strip()])
    if fields.get("rpo_steps", {}).get("value"):
        checklist_items.extend([x.strip() for x in str(fields["rpo_steps"]["value"]).split("\n") if x.strip()])
    if fields.get("known_failure_scenarios", {}).get("value"):
        warnings.extend([x.strip() for x in str(fields["known_failure_scenarios"]["value"]).split("\n") if x.strip()])
    if fields.get("recovery_contact", {}).get("value"):
        paragraphs.append(f"Recovery contact: {fields['recovery_contact']['value']}")

    blocks = []
    if checklist_items:
        blocks.append(build_checklist_block("Recovery actions", checklist_items))
    if warnings:
        blocks.append(build_warning_block("Known failure scenarios", warnings))
    if paragraphs:
        blocks.insert(0, build_narrative_block(title, paragraphs))

    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
