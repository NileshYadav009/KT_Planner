from typing import Dict, Any, List, Optional
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.narrative import build_block as build_narrative_block


def _get_structured(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    structured = section.get("_structured")
    return structured if isinstance(structured, dict) else None


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
    rows = _coverage_rows(section)
    blocks = []

    structured = _get_structured(section)
    if structured is not None:
        issues = structured.get("issues") or []
        rows = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            rows.append({
                "Symptom": issue.get("issue", ""),
                "Cause": issue.get("cause", ""),
                "Fix": issue.get("fix", ""),
            })
        if rows:
            blocks.append(build_decision_table("Failure symptoms and remediation", ["Symptom", "Cause", "Fix"], rows))
            return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}

    if rows:
        blocks.append(build_decision_table("Failure symptoms and remediation", ["Symptom", "Cause", "Fix"], rows))
    else:
        content = section.get("coverage_content") or []
        if isinstance(content, str):
            content = [content]
        blocks.append(build_narrative_block(title, [str(item).strip() for item in content if str(item).strip()]))

    if not blocks or not rows:
        if not blocks:
            blocks.append(build_narrative_block(title, ["Common failures are being characterized from extracted coverage."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
