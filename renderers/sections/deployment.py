import re
from typing import Dict, Any, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.checklist import build_block as build_checklist_block
from renderers.blocks.table import build_block as build_decision_table, parse_table_rows
from renderers.blocks.timeline import build_block as build_timeline_block
from renderers.blocks.common import no_coverage_block


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    return [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]


def _normalized_key(text: str) -> str:
    return " ".join(str(text or "").split()).lower()[:120]


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

    # Fields populated independently (pattern/semantic/LLM extraction, see
    # field_populator.py) can end up carrying the same broad matched text —
    # e.g. a short transcript's deployment_window/pre_deployment_checks/
    # post_deployment_validation all resolving to the same block. Dedup by
    # normalized text across everything this renderer emits so the same
    # content doesn't appear 2-3 times in the PDF.
    seen_keys = set()

    def _claim(text: str) -> bool:
        key = _normalized_key(text)
        if not key or key in seen_keys:
            return False
        seen_keys.add(key)
        return True

    if fields.get("deployment_steps", {}).get("value"):
        deployment_value = fields["deployment_steps"]["value"]
        _claim(str(deployment_value))
        if "|" in str(deployment_value):
            decision_rows = parse_table_rows(deployment_value, ["Step", "Action", "Tool/Command", "Expected Result"])
        else:
            checklist = [line.strip() for line in str(deployment_value).split("\n") if line.strip()]

    if not checklist and not decision_rows and coverage_content:
        for item in coverage_content:
            text = str(item).strip()
            if not text:
                continue
            if re.match(r"^\d+\.", text) or any(kw in text.lower() for kw in ["merge", "trigger", "deploy", "rollback", "helm", "release", "gitops"]):
                checklist.append(text)

    if fields.get("rollback_scenarios", {}).get("value"):
        rollback_rows = parse_table_rows(fields["rollback_scenarios"]["value"], ["Scenario", "Rollback Action", "Risk Level"])

    if fields.get("deployment_window", {}).get("value") and _claim(str(fields["deployment_window"]["value"])):
        timeline.append({"label": "Deployment window", "description": fields["deployment_window"]["value"]})
    if fields.get("pre_deployment_checks", {}).get("value") and _claim(str(fields["pre_deployment_checks"]["value"])):
        paragraphs.append(f"Pre-deployment checks: {fields['pre_deployment_checks']['value']}")
    if fields.get("post_deployment_validation", {}).get("value") and _claim(str(fields["post_deployment_validation"]["value"])):
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
            blocks.append(no_coverage_block(title))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
