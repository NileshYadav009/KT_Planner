"""
schema_generator.py
===================
Generates a coverage-aware dynamic schema for KT field population.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

TECH_STACK_FIELD_ADDITIONS = {
    r"\b(redis|elasticache)\b": {
        "section": "system_overview",
        "field": {
            "id": "cache_layer",
            "label": "Cache Layer",
            "type": "text",
            "description": "Redis / ElastiCache configuration and sizing",
            "required": False,
            "dynamic": True,
        },
    },
    r"\b(kafka|kinesis|event\s+stream)\b": {
        "section": "system_overview",
        "field": {
            "id": "event_streaming",
            "label": "Event Streaming Platform",
            "type": "text",
            "description": "Kafka topics, consumer groups, retention policy",
            "required": False,
            "dynamic": True,
        },
    },
    r"\b(vault|secret\s+management)\b": {
        "section": "deployment_and_rollback",
        "field": {
            "id": "secret_management",
            "label": "Secret Management",
            "type": "text",
            "description": "Vault paths, rotation policy, access patterns",
            "required": False,
            "dynamic": True,
        },
    },
    r"\b(argocd|argo\s+cd|gitops)\b": {
        "section": "deployment_and_rollback",
        "field": {
            "id": "gitops_tool",
            "label": "GitOps Tool",
            "type": "text",
            "description": "ArgoCD app name, sync policy, target cluster",
            "required": False,
            "dynamic": True,
        },
    },
    r"\b(trivy|sonarqube|veracode|security\s+scan)\b": {
        "section": "security_controls",
        "field": {
            "id": "security_scan_config",
            "label": "Security Scan Configuration",
            "type": "text",
            "description": "Scanner tool, severity thresholds, gate conditions",
            "required": False,
            "dynamic": True,
        },
    },
    r"\b(pagerduty|opsgenie|on[\s-]call)\b": {
        "section": "ownership_escalation",
        "field": {
            "id": "oncall_tool",
            "label": "On-Call Tool & Schedule",
            "type": "text",
            "description": "PagerDuty service name, rotation schedule, escalation policy",
            "required": False,
            "dynamic": True,
        },
    },
    r"\b(terraform)\b": {
        "section": "deployment_and_rollback",
        "field": {
            "id": "terraform_workspace",
            "label": "Terraform Workspace / State Location",
            "type": "text",
            "description": "State backend, workspace name, key paths",
            "required": False,
            "dynamic": True,
        },
    },
    r"\b(confluence|wiki|runbook)\b": {
        "section": "architecture_reference",
        "field": {
            "id": "documentation_links",
            "label": "Documentation Links",
            "type": "text",
            "description": "Confluence space, runbook URLs, wiki links",
            "required": False,
            "dynamic": True,
        },
    },
}

OPTIONAL_SECTION_INCLUSION_RULES = {
    "plain_english_notes": 0.30,
    "cost_optimization": 0.35,
    "security_controls": 0.40,
    "disaster_recovery": 0.40,
    "first_30_day_ownership": 0.25,
    "handover_completion": 0.10,
}


def _get_combined_text(coverage_entry: Dict[str, Any]) -> str:
    parts = []
    content = coverage_entry.get("content", [])
    if isinstance(content, list):
        parts.extend(content)
    elif isinstance(content, str):
        parts.append(content)
    for sentence in coverage_entry.get("sentences", []):
        if isinstance(sentence, dict):
            text = sentence.get("text", "")
            if text:
                parts.append(text)
    return " ".join(parts).lower()


def generate_dynamic_schema(
    coverage: Dict[str, Any],
    base_schema: List[Dict],
    include_missing_required: bool = True,
) -> List[Dict]:
    dynamic_sections = []
    all_text = " ".join(_get_combined_text(cov) for cov in coverage.values())

    dynamic_field_additions: Dict[str, List[Dict]] = {}
    for pattern, config in TECH_STACK_FIELD_ADDITIONS.items():
        if re.search(pattern, all_text, re.IGNORECASE):
            section_id = config["section"]
            dynamic_field_additions.setdefault(section_id, []).append(config["field"])
            logger.debug("Dynamic field added to %s: %s", section_id, config["field"]["id"])

    for section in base_schema:
        section_id = section.get("id")
        is_required = section.get("required", False)
        cov_entry = coverage.get(section_id, {})
        status = cov_entry.get("status", "missing")
        confidence = cov_entry.get("confidence", 0.0) or 0.0

        include = False
        if is_required and include_missing_required:
            include = True
        elif status in ("covered", "weak"):
            include = True
        elif section_id in OPTIONAL_SECTION_INCLUSION_RULES:
            min_conf = OPTIONAL_SECTION_INCLUSION_RULES[section_id]
            include = (confidence >= min_conf) or (status != "missing")

        if not include:
            logger.debug("Section excluded from dynamic schema: %s (status=%s)", section_id, status)
            continue

        dynamic_section = {**section}
        if section_id in dynamic_field_additions:
            existing_fields = list(dynamic_section.get("fields", []) or [])
            existing_ids = {field.get("id") for field in existing_fields}
            for extra_field in dynamic_field_additions[section_id]:
                if extra_field["id"] not in existing_ids:
                    existing_fields.append(extra_field)
            dynamic_section["fields"] = existing_fields

        dynamic_section["_coverage_status"] = status
        dynamic_section["_coverage_confidence"] = round(confidence, 3)
        dynamic_sections.append(dynamic_section)

    logger.info("Dynamic schema: %d/%d sections included", len(dynamic_sections), len(base_schema))
    return dynamic_sections
