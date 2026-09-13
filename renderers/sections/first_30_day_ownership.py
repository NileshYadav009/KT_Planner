from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.ownership import build_block as build_ownership_table
from renderers.blocks.common import no_coverage_block

# kt_schema_new.json's week1..week4 fields (see field_populator.py) — the
# schema-authored labels are the fallback when a field's own label wasn't
# threaded through into the knowledge object.
_WEEK_FIELD_LABELS = {"week1": "Week 1", "week2": "Week 2", "week3": "Week 3", "week4": "Week 4"}


def _field_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    """Read the real per-week values populate_fields() extracted, instead
    of the pipe-delimited coverage_content parsing below — natural speech
    essentially never contains a literal "|", so that fallback always
    produced a single row with a blank second column regardless of how
    much real per-week content had actually been captured."""
    fields = section.get("fields") or {}
    rows = []
    for field_id, fallback_label in _WEEK_FIELD_LABELS.items():
        entry = fields.get(field_id) or {}
        value = entry.get("value")
        if isinstance(value, str) and value.strip():
            rows.append({"role": entry.get("label") or fallback_label, "team": value.strip()})
    return rows


def _coverage_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    rows = []
    for item in content:
        text = str(item).strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if len(parts) >= 2:
            rows.append({"role": parts[0], "team": parts[1]})
        else:
            rows.append({"role": text, "team": ""})
    return rows


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "First 30-Day Ownership")
    rows = _field_rows(section) or _coverage_rows(section)
    blocks = []

    if rows:
        blocks.append(build_ownership_table("Ownership handover", rows))
    else:
        content = section.get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        paragraphs = [str(item).strip() for item in content if str(item).strip()]
        if paragraphs:
            blocks.append(build_narrative_block(title, paragraphs))

    if not blocks:
        blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
