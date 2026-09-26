from typing import Any, Dict, List
from renderers.blocks.narrative import build_block as build_narrative_block
from renderers.blocks.technology_grid import build_block as build_technology_grid
from renderers.blocks.common import (
    no_coverage_block,
    coverage_paragraphs as shared_coverage_paragraphs,
)

# Generic tool -> technology-category mapping, keyed by lowercased tool name
# as matched by field_populator.py's PATTERN_EXTRACTORS["tools"] regex. Not
# specific to any one transcript's stack choices — extend this table (not
# per-transcript logic) when a new common tool needs a home.
_TECH_CATEGORY_MAP = {
    "react": "Frontend", "angular": "Frontend", "vue": "Frontend", "vue.js": "Frontend",
    "fastapi": "Backend", "fast api": "Backend", "django": "Backend", "flask": "Backend",
    "node": "Backend", "node.js": "Backend", "express": "Backend",
    ".net": "Backend", "asp.net": "Backend", "dotnet": "Backend",
    "postgresql": "Database", "mysql": "Database", "mongodb": "Database",
    "amazon rds": "Database",
    "redis": "Cache",
    "amazon eks": "Compute", "kubernetes": "Compute", "docker": "Compute", "rancher": "Compute",
    "cloudfront": "Edge / ingress", "application load balancer": "Edge / ingress",
    "alb": "Edge / ingress", "load balancer": "Edge / ingress",
    "terraform": "Infrastructure", "ansible": "Infrastructure",
    "argocd": "GitOps / deployment", "flux": "GitOps / deployment",
    "jenkins": "GitOps / deployment", "gitlab ci": "GitOps / deployment",
    "github actions": "GitOps / deployment", "helm": "GitOps / deployment",
    "vault": "Secrets", "consul": "Secrets",
    "trivy": "Security", "sonarqube": "Security", "veracode": "Security",
    "amazon ecr": "Security", "s3": "Storage",
    "prometheus": "Observability", "grafana": "Observability", "cloudwatch": "Observability",
    "datadog": "Observability", "splunk": "Observability", "elk": "Observability",
    "elasticsearch": "Observability", "logstash": "Observability", "kibana": "Observability",
    "kafka": "Messaging", "rabbitmq": "Messaging",
    "pagerduty": "Alerting", "opsgenie": "Alerting",
    "nexus": "Artifact management", "artifactory": "Artifact management",
    # AWS equivalents of the same roles (the acronym forms are what
    # knowledge_builder.py's _CANONICAL_TERM_ALIASES collapses to).
    "amazon sqs": "Messaging", "sqs": "Messaging",
    "amazon ecr": "Container registry", "ecr": "Container registry",
    "rds": "Database",
    # The rest of a mainstream AWS stack. Without these the tools regex can
    # recognise the term while _categorize_technologies() still drops it (it
    # skips anything with no category), so a real AWS KT naming Aurora, MSK,
    # OpenSearch and Route 53 rendered a Technology Summary missing its
    # database, messaging, logging and DNS tiers entirely -- the same defect
    # already fixed once for Azure, re-found on the AWS side.
    "eks": "Compute",
    "aurora": "Database", "aurora postgresql": "Database", "amazon aurora": "Database",
    "amazon msk": "Messaging", "msk": "Messaging",
    "opensearch": "Observability", "amazon opensearch": "Observability",
    "route 53": "Edge / ingress", "route53": "Edge / ingress",
    "aws waf": "Security", "waf": "Security",
    "aws kms": "Security", "kms": "Security",
    "spring boot": "Backend", "spring": "Backend",
    # Azure. Without these the entire Azure stack was silently dropped from
    # the Technology summary -- _categorize_technologies() skips any tool
    # with no category, so a real Azure KT listing 18 identified components
    # rendered only 3 rows (Frontend/Observability/Alerting). Confirmed on
    # a live generated PDF.
    "azure kubernetes service": "Compute", "aks": "Compute",
    "azure sql": "Database",
    "azure cache for redis": "Cache",
    "azure service bus": "Messaging", "service bus": "Messaging",
    "azure front door": "Edge / ingress", "application gateway": "Edge / ingress",
    "azure container registry": "Container registry", "acr": "Container registry",
    "azure devops": "GitOps / deployment", "flux": "GitOps / deployment",
    "bicep": "Infrastructure",
    "azure key vault": "Secrets", "key vault": "Secrets",
    "azure monitor": "Observability", "application insights": "Observability",
    "azure blob storage": "Storage", "blob storage": "Storage",
    # GCP equivalents of the same roles.
    "google kubernetes engine": "Compute", "gke": "Compute",
    "bigquery": "Database", "cloud sql": "Database", "firestore": "Database",
    "bigtable": "Database", "cloud spanner": "Database",
    "memorystore": "Cache",
    "pub/sub": "Messaging",
    "artifact registry": "Container registry", "container registry": "Container registry",
    "secret manager": "Secrets",
    "cloud monitoring": "Observability", "cloud logging": "Observability",
    "stackdriver": "Observability",
    "cloud build": "GitOps / deployment",
    "dataflow": "Data processing", "airflow": "Workflow orchestration",
    "vertex ai": "Machine learning",
    "cloud storage": "Storage",
}


def _drop_generic_duplicates(values: List[str]) -> List[str]:
    """Remove a generic term when the branded product it names is also
    listed in the same category, e.g. "Azure Kubernetes Service; Kubernetes"
    -> "Azure Kubernetes Service". A transcript naturally says the full
    product name once and the generic word afterwards, and both then match
    the tools regex, so the summary listed the same tier twice.
    """
    kept: List[str] = []
    for value in values:
        lowered = value.lower()
        if any(lowered != other.lower() and lowered in other.lower() for other in values):
            continue
        kept.append(value)
    return kept or values


def _categorize_technologies(key_technologies_value: str) -> List[Dict[str, str]]:
    tools = [t.strip() for t in key_technologies_value.split(",") if t.strip()]
    by_category: Dict[str, List[str]] = {}
    for tool in tools:
        category = _TECH_CATEGORY_MAP.get(tool.lower())
        if not category:
            continue
        by_category.setdefault(category, []).append(tool)
    return [
        {"label": category, "value": "; ".join(_drop_generic_duplicates(values))}
        for category, values in by_category.items()
    ]


def _coverage_paragraphs(section: Dict[str, Any]) -> List[str]:
    # Shared implementation: splits polish-pass bullet blobs and drops
    # repeats. See renderers/blocks/common.coverage_paragraphs().
    return shared_coverage_paragraphs(section)


def _field_value(fields: Dict[str, Any], *path: str):
    node: Any = fields
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if isinstance(node, dict):
        return node.get("value")
    return None


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    title = section.get("title", "System Overview")
    fields = section.get("fields", {})

    knowledge_rows: List[Dict[str, str]] = []

    def _add(label: str, value: Any):
        if isinstance(value, str) and value.strip():
            knowledge_rows.append({"label": label, "value": value.strip()})

    _add("Business criticality", _field_value(fields, "business_criticality"))
    _add("Business volume", _field_value(fields, "orders_per_day"))
    _add("Customer impact", _field_value(fields, "impact_if_down", "what_breaks"))
    _add("Operational impact", _field_value(fields, "impact_if_down", "who_affected"))
    _add("Customer reach", _field_value(fields, "customer_reach"))
    _add("What it does", _field_value(fields, "system_in_5_lines", "what_it_does"))
    _add("Business purpose", _field_value(fields, "business_purpose", "problem_solved"))
    _add("Why the business depends on it", _field_value(fields, "business_purpose", "business_dependency"))

    tech_rows: List[Dict[str, str]] = []
    key_technologies = _field_value(fields, "key_technologies")
    if isinstance(key_technologies, str) and key_technologies.strip():
        tech_rows = _categorize_technologies(key_technologies)

    # Dynamically-added fields (schema_generator.py:TECH_STACK_FIELD_ADDITIONS)
    # for content that implies a cache layer / event-streaming platform —
    # not always present, only added when the transcript's tech mix triggers
    # the relevant pattern.
    for label, field_id in (("Cache Layer", "cache_layer"), ("Event Streaming", "event_streaming")):
        value = _field_value(fields, field_id)
        if isinstance(value, str) and value.strip():
            tech_rows.append({"label": label, "value": value.strip()})

    blocks = []
    if knowledge_rows:
        blocks.append(build_technology_grid("Captured knowledge", knowledge_rows))
    if tech_rows:
        blocks.append(build_technology_grid("Technology summary", tech_rows))

    if not blocks:
        fallback = _coverage_paragraphs(section)
        if fallback:
            blocks.append(build_narrative_block(title, fallback))
        else:
            blocks.append(no_coverage_block(title))
    else:
        # Anything captured for this section that didn't map to one of the
        # explicit fields above (e.g. free-form context) should still
        # surface, not silently vanish just because some fields matched.
        used_texts = {row["value"] for row in knowledge_rows}
        leftover = [p for p in _coverage_paragraphs(section) if p not in used_texts]
        if leftover:
            blocks.append(build_narrative_block("Additional context", leftover))

    return {"section_id": section.get("id"), "section_title": title, "blocks": blocks}
