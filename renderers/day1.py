from typing import Dict, Any, List
from renderers.base import build_checklist_block, build_narrative_block


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
        blocks.append(build_narrative_block(title, ["First-day guidance is being assembled from KT evidence."]))
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
