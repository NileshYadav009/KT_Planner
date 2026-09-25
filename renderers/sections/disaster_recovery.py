from typing import Dict, Any, List
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.warning import build_block as build_warning_block
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.common import (
    no_coverage_block,
    coverage_paragraphs as shared_coverage_paragraphs,
)


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    # Shared implementation: splits polish-pass bullet blobs and drops
    # repeats. See renderers/blocks/common.coverage_paragraphs().
    return shared_coverage_paragraphs(section)


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
    if fields.get("dr_testing_frequency", {}).get("value"):
        paragraphs.append(f"DR testing frequency: {fields['dr_testing_frequency']['value']}")
    # rto_metric/rpo_metric (field_populator.extract_rto_rpo) are a plain
    # duration, deliberately kept out of rto_steps/rpo_steps above (which
    # hold the LLM's own recovery PROCEDURE narrative, a different fact) --
    # rendered as their own clearly labeled lines so "2 hours" never appears
    # bare with nothing saying what it measures.
    if fields.get("rto_metric", {}).get("value"):
        paragraphs.append(f"RTO (Recovery Time Objective): {fields['rto_metric']['value']}")
    if fields.get("rpo_metric", {}).get("value"):
        paragraphs.append(f"RPO (Recovery Point Objective): {fields['rpo_metric']['value']}")

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
