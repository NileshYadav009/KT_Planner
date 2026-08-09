from typing import Dict, Any, List
from renderers.base import (
    build_narrative_block,
    build_checklist_block,
    build_deployment_timeline,
    build_decision_table,
)


def _parse_table_rows(value: Any, columns: List[str]) -> List[Dict[str, str]]:
    text = str(value or "")
    rows = []
    for line in [line.strip() for line in text.split("\n") if line.strip()]:
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|")]
            row = {columns[idx]: cells[idx] if idx < len(cells) else "" for idx in range(len(columns))}
            rows.append(row)
        else:
            rows.append({columns[0]: line} if columns else {"Step": line})
    return rows


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Deployment & Rollback")
    fields = section.get("fields", {})

    paragraphs = []
    checklist = []
    timeline: List[Dict[str, str]] = []
    decision_rows: List[Dict[str, str]] = []
    rollback_rows: List[Dict[str, str]] = []

    if fields.get("deployment_steps", {}).get("value"):
        deployment_value = fields["deployment_steps"]["value"]
        if "|" in str(deployment_value):
            decision_rows = _parse_table_rows(deployment_value, ["Step", "Action", "Tool/Command", "Expected Result"])
        else:
            checklist = [line.strip() for line in str(deployment_value).split("\n") if line.strip()]

    if fields.get("rollback_scenarios", {}).get("value"):
        rollback_rows = _parse_table_rows(
            fields["rollback_scenarios"]["value"],
            ["Scenario", "Rollback Action", "Risk Level"],
        )

    if fields.get("deployment_window", {}).get("value"):
        timeline.append({"label": "Deployment window", "description": fields["deployment_window"]["value"]})

    if fields.get("pre_deployment_checks", {}).get("value"):
        paragraphs.append(f"Pre-deployment checks: {fields['pre_deployment_checks']['value']}")

    if fields.get("post_deployment_validation", {}).get("value"):
        paragraphs.append(f"Post-deployment validation: {fields['post_deployment_validation']['value']}")

    blocks = []
    if paragraphs:
        blocks.append(build_narrative_block(title, paragraphs))
    if decision_rows:
        blocks.append(build_decision_table("Deployment steps", ["Step", "Action", "Tool/Command", "Expected Result"], decision_rows))
    elif checklist:
        blocks.append(build_checklist_block("Deployment steps", checklist))
    if rollback_rows:
        blocks.append(build_decision_table("Rollback scenarios", ["Scenario", "Rollback Action", "Risk Level"], rollback_rows))
    if timeline:
        blocks.append(build_deployment_timeline("Deployment timeline", timeline))
    if not blocks:
        blocks.append(build_narrative_block(title, ["Deployment and rollback guidance is being assembled."]))
    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
