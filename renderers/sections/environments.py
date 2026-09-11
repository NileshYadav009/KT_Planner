from typing import Any, Dict, List
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.common import no_coverage_block

COLUMNS = ["Environment", "Known characteristics"]

NOT_PROVIDED_NOTE = (
    "Exact environment names, URLs, AWS accounts, regions, access procedures "
    "and namespaces were not covered — do not infer them."
)


def _field_value(fields: Dict[str, Any], field_id: str):
    entry = fields.get(field_id)
    if isinstance(entry, dict):
        return entry.get("value")
    return None


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    return [str(item).strip() for item in content if isinstance(item, str) and item.strip()]


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Environments")
    fields = section.get("fields", {})

    rows = []
    for label, field_id in (
        ("Production", "production_notes"),
        ("Staging", "staging_notes"),
        ("Non-production", "non_production_notes"),
    ):
        value = _field_value(fields, field_id)
        if isinstance(value, str) and value.strip():
            rows.append({"Environment": label, "Known characteristics": value.strip()})

    blocks = []
    if rows:
        blocks.append(build_decision_table("Environment knowledge", COLUMNS, rows))
        known_differences = _field_value(fields, "known_differences")
        if isinstance(known_differences, str) and known_differences.strip():
            blocks.append(build_narrative_block("Known differences / limitations", [known_differences.strip()]))
        blocks.append(build_narrative_block("Do not over-infer", [NOT_PROVIDED_NOTE]))
        return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}

    fallback = _coverage_paragraphs(section)
    if fallback:
        blocks.append(build_narrative_block(title, fallback))
        blocks.append(build_narrative_block("Do not over-infer", [NOT_PROVIDED_NOTE]))
    else:
        blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
