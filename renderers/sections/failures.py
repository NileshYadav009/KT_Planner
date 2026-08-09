from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.warning import build_block as build_warning_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Common Failures")
    fields = section.get("fields", {})

    paragraphs: List[str] = []
    warnings: List[str] = []

    if fields.get("common_failures", {}).get("value"):
        warnings.append(fields["common_failures"]["value"])
    if fields.get("known_bad_days", {}).get("value"):
        paragraphs.append(f"Known bad days: {fields['known_bad_days']['value']}")

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if warnings:
        blocks.append(build_warning_block("Failure warnings", warnings))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(build_narrative_block(title, ["Failure modes are being identified from knowledge evidence."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
