import re
from typing import Dict, Any, List, Optional
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.table import build_block as build_decision_table
from renderers.blocks.timeline import build_block as build_timeline_block


def _get_structured(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    structured = section.get("_structured")
    return structured if isinstance(structured, dict) else None


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def _parse_table_rows(value: Any, columns: List[str]) -> List[Dict[str, str]]:
    text = str(value or "")
    rows: List[Dict[str, str]] = []
    for line in [line.strip() for line in text.split("\n") if line.strip()]:
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|")]
            rows.append({columns[idx]: cells[idx] if idx < len(cells) else "" for idx in range(len(columns))})
        else:
            rows.append({columns[0]: line} if columns else {"Step": line})
    return rows


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Deployment & Rollback")
    fields = section.get("fields", {})
    coverage_content = section.get("coverage_content", [])
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]

    paragraphs: List[str] = []
    checklist: List[str] = []
    decision_rows: List[Dict[str, str]] = []
    rollback_rows: List[Dict[str, str]] = []
    timeline: List[Dict[str, str]] = []

    structured = _get_structured(section)
    if structured is not None:
        steps = structured.get("deployment_steps") or []
        if isinstance(steps, list) and steps:
            checklist = [str(step).strip() for step in steps if isinstance(step, str) and step.strip()]
        trigger = structured.get("trigger")
        window = structured.get("window")
        approver = structured.get("approver")
        duration = structured.get("duration")
        rollback = structured.get("rollback") or {}

        if trigger:
            paragraphs.append(f"Trigger: {trigger}")
        if window:
            paragraphs.append(f"Window: {window}")
        if approver:
            paragraphs.append(f"Approver: {approver}")
        if duration:
            paragraphs.append(f"Duration: {duration}")
        if rollback and isinstance(rollback, dict):
            rollback_trigger = rollback.get("trigger")
            rollback_action = rollback.get("action")
            rollback_target = rollback.get("target_time")
            rollback_lines = []
            if rollback_trigger:
                rollback_lines.append(f"Rollback trigger: {rollback_trigger}")
            if rollback_action:
                rollback_lines.append(f"Rollback action: {rollback_action}")
            if rollback_target:
                rollback_lines.append(f"Target time: {rollback_target}")
            if rollback_lines:
                paragraphs.append("Rollback:")
                paragraphs.extend(rollback_lines)

    if not checklist and not decision_rows and coverage_content:
        for item in coverage_content:
            text = str(item).strip()
            if not text:
                continue
            if re.match(r"^\d+\.", text) or any(kw in text.lower() for kw in ["merge", "trigger", "deploy", "rollback", "helm", "release", "gitops"]):
                checklist.append(text)

    if fields.get("deployment_steps", {}).get("value"):
        deployment_value = fields["deployment_steps"]["value"]
        if "|" in str(deployment_value):
            decision_rows = _parse_table_rows(deployment_value, ["Step", "Action", "Tool/Command", "Expected Result"])
        else:
            checklist = [line.strip() for line in str(deployment_value).split("\n") if line.strip()]

    if fields.get("rollback_scenarios", {}).get("value"):
        rollback_rows = _parse_table_rows(fields["rollback_scenarios"]["value"], ["Scenario", "Rollback Action", "Risk Level"])

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
        blocks.append(build_timeline_block("Deployment timeline", timeline))
    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(build_narrative_block(title, ["Deployment and rollback guidance is being assembled."]))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
