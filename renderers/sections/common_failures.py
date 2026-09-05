from typing import Dict, Any, List
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.common import no_coverage_block

# Matches kt_schema_new.json's declared columns for the common_failures table
# section. The legacy pipe-parsed fallback below only ever produces 3 of these
# (Symptom/Cause/Fix) since that format never carried frequency/ticket data.
STRUCTURED_COLUMNS = ["Issue/Symptom", "Likely Cause", "How to Fix", "Frequency", "KEDB / Ticket Link"]


def _structured_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    failures = (section.get("_structured") or {}).get("failures")
    if not isinstance(failures, list):
        return []
    rows = []
    for item in failures:
        if not isinstance(item, dict):
            continue
        symptom = str(item.get("symptom") or "").strip()
        if not symptom:
            continue
        rows.append({
            "Issue/Symptom": symptom,
            "Likely Cause": str(item.get("cause") or "").strip(),
            "How to Fix": str(item.get("fix") or "").strip(),
            "Frequency": str(item.get("frequency") or "").strip(),
            "KEDB / Ticket Link": str(item.get("ticket") or "").strip(),
        })
    return rows


def _coverage_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    failures = section.get("coverage_content") or []
    if isinstance(failures, str):
        failures = [failures]

    rows = []
    for item in failures:
        text = str(item).strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if len(parts) >= 3:
            rows.append({"Symptom": parts[0], "Cause": parts[1], "Fix": parts[2]})
        elif len(parts) == 2:
            rows.append({"Symptom": parts[0], "Cause": parts[1], "Fix": ""})
        else:
            rows.append({"Symptom": text, "Cause": "", "Fix": ""})
    return rows


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Common Failures")
    structured_rows = _structured_rows(section)
    blocks = []

    if structured_rows:
        blocks.append(build_decision_table("Failure symptoms and remediation", STRUCTURED_COLUMNS, structured_rows))
        return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}

    rows = _coverage_rows(section)

    if rows:
        blocks.append(build_decision_table("Failure symptoms and remediation", ["Symptom", "Cause", "Fix"], rows))
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
