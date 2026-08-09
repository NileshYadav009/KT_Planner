from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.checklist import build_block as build_checklist_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Day 1 Survival Checklist")
    fields = section.get("fields", {})

    checklist: List[str] = []
    paragraphs: List[str] = []

    if fields.get("safe_first_actions", {}).get("value"):
        checklist.extend([line.strip() for line in str(fields["safe_first_actions"]["value"]).split("\n") if line.strip()])
    if fields.get("access_to_request", {}).get("value"):
        paragraphs.append(f"Access to request: {fields['access_to_request']['value']}")
    if fields.get("dont_do_day1", {}).get("value"):
        checklist.extend([f"Do NOT: {line.strip()}" for line in str(fields["dont_do_day1"]["value"]).split("\n") if line.strip()])

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if checklist:
        blocks.append(build_checklist_block(title, checklist))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(build_narrative_block(title, ["First-day guidance is being assembled from KT evidence."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
