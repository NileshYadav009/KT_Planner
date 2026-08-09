from typing import Any, Dict, List, Optional


def build_narrative_block(title: str, paragraphs: List[str]) -> Dict[str, Any]:
    return {
        "type": "NarrativeBlock",
        "title": title,
        "paragraphs": [p for p in paragraphs if p],
    }


def build_checklist_block(title: str, items: List[str]) -> Dict[str, Any]:
    return {
        "type": "ChecklistBlock",
        "title": title,
        "items": [item for item in items if item],
    }


def build_warning_block(title: str, warnings: List[str]) -> Dict[str, Any]:
    return {
        "type": "WarningBlock",
        "title": title,
        "warnings": [w for w in warnings if w],
    }


def build_technology_grid(title: str, rows: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "type": "TechnologyGrid",
        "title": title,
        "rows": [row for row in rows if row.get("label") and row.get("value")],
    }


def build_deployment_timeline(title: str, entries: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "type": "DeploymentTimeline",
        "title": title,
        "entries": [entry for entry in entries if entry.get("label") and entry.get("description")],
    }


def build_ownership_table(title: str, rows: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "type": "OwnershipTable",
        "title": title,
        "rows": [row for row in rows if row.get("role") or row.get("team")],
    }


def build_decision_table(title: str, columns: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "DecisionTable",
        "title": title,
        "columns": [col for col in columns if col],
        "rows": [row for row in rows if isinstance(row, dict)],
    }


def build_troubleshooting_block(title: str, steps: List[str]) -> Dict[str, Any]:
    return {
        "type": "TroubleshootingBlock",
        "title": title,
        "steps": [step for step in steps if step],
    }


def build_code_block(title: str, code: str, language: str = "text") -> Dict[str, Any]:
    return {
        "type": "CodeBlock",
        "title": title,
        "code": code,
        "language": language,
    }


def build_default_block(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "Section")
    paragraphs = []
    for field in section.get("fields", {}).values():
        value = field.get("value")
        if value:
            paragraphs.append(f"{field.get('label', field.get('id', 'Field'))}: {value}")

    if not paragraphs:
        coverage_content = section.get("coverage_content") or []
        if isinstance(coverage_content, str):
            coverage_content = [coverage_content]
        paragraphs = [str(item).strip() for item in coverage_content if isinstance(item, str) and item.strip()]

    if not paragraphs:
        paragraphs = ["No rendered content is available for this section."]
    return build_narrative_block(title, paragraphs)
