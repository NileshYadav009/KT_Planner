"""
architecture_diagram.py
========================
Builds a simple, generic top-down "mental model" diagram from a flat list of
detected architecture component names (schema_generator/knowledge_builder's
tools regex output): the main request-flow chain (Customer -> ... -> compute
-> {service, database, cache, queue}), plus separate supporting-
infrastructure flows (CI/CD pipeline, IaC, secrets, observability,
alerting) shown on their own rather than force-fit into the request path.

Deliberately NOT a general-purpose relationship extractor that parses
arbitrary transcript sentences for "X connects to Y" phrasing — that's
fragile and hard to generalize correctly across arbitrary transcripts.
Instead this uses a coarse, extensible layer classification for known
component names, and renders the shape those layers imply. Only layers
that were actually named in the transcript are drawn — nothing is invented
to fill in a "typical" architecture.
"""
from collections import defaultdict
from typing import Dict, List, Optional

# Maps a canonical component name (lowercased, as PATTERN_EXTRACTORS["tools"]
# captures it in field_populator.py, after knowledge_builder.py's
# acronym-to-branded-name canonicalization) to a coarse architectural layer.
_LAYER_TERMS: Dict[str, List[str]] = {
    "frontend": ["react", "angular", "vue", "vue.js"],
    "cdn": ["cloudfront", "azure front door"],
    "load_balancer": ["application load balancer", "alb", "load balancer", "application gateway"],
    "compute": ["amazon eks", "kubernetes", "docker", "rancher", "azure kubernetes service", "aks"],
    "service": ["fastapi", "fast api", "django", "flask", "node.js", "node", "express"],
    "database": ["postgresql", "mysql", "mongodb", "amazon rds", "azure sql"],
    "cache": ["redis"],
    "queue": ["kafka", "rabbitmq", "sqs", "amazon sqs", "azure service bus", "service bus"],
    "registry": ["amazon ecr", "azure container registry", "acr"],
    "cicd": ["github actions", "gitlab ci", "jenkins", "azure devops"],
    "gitops": ["argocd", "flux"],
    "iac": ["terraform", "bicep", "ansible"],
    "secrets": ["vault", "azure key vault", "key vault", "consul"],
    "monitoring": [
        "prometheus", "grafana", "cloudwatch", "datadog", "splunk", "elk",
        "elasticsearch", "logstash", "kibana", "azure monitor", "application insights",
    ],
    "alerting": ["pagerduty", "opsgenie"],
}

# The main top-down request-flow chain, in order.
_CHAIN_LAYERS = ["frontend", "cdn", "load_balancer", "compute"]
# Layers that fan out FROM the compute hub, rather than continuing the chain.
_FANOUT_LAYERS = ["service", "database", "cache", "queue"]

_ROOT_LABEL = "Customer"


def _classify(component: str) -> Optional[str]:
    lowered = component.strip().lower()
    for layer, names in _LAYER_TERMS.items():
        if lowered in names:
            return layer
    return None


def _group_by_layer(components: List[str]) -> Dict[str, List[str]]:
    """Every detected component, grouped by layer, preserving the
    transcript-order each component was first seen in (components is
    already deduplicated by the time it reaches here)."""
    grouped: Dict[str, List[str]] = defaultdict(list)
    for comp in components or []:
        layer = _classify(comp)
        if layer:
            grouped[layer].append(comp)
    return grouped


def _vertical_chain(nodes: List[str]) -> str:
    lines: List[str] = []
    for i, node in enumerate(nodes):
        lines.append(node)
        if i < len(nodes) - 1:
            lines.append("   │")
            lines.append("   ▼")
    return "\n".join(lines)


def _build_main_flow(by_layer: Dict[str, List[str]]) -> Optional[str]:
    """The primary request-flow tree: Customer -> chain layers -> fan-out
    from the compute hub. Falls back to a flat chain (no fan-out) when no
    compute/orchestration hub was actually named, rather than fabricating a
    branch point the transcript never stated.
    """
    chain = [by_layer[layer][0] for layer in _CHAIN_LAYERS if layer in by_layer]
    fanout = [by_layer[layer][0] for layer in _FANOUT_LAYERS if layer in by_layer]

    if not chain and not fanout:
        return None

    has_hub = "compute" in by_layer
    if not has_hub and fanout:
        chain = chain + fanout
        fanout = []

    lines: List[str] = []
    nodes = ([_ROOT_LABEL] if chain else []) + chain
    for i, node in enumerate(nodes):
        lines.append(node)
        if i < len(nodes) - 1:
            lines.append("   │")
            lines.append("   ▼")
        elif fanout:
            # Transitioning into a fan-out (branch), not a single next step
            # — just a connecting line, no "▼" (which implies flowing into
            # one node directly below, not branching into several).
            lines.append("   │")

    if fanout:
        if not nodes:
            lines.append(_ROOT_LABEL)
        for j, child in enumerate(fanout):
            connector = "└──" if j == len(fanout) - 1 else "├──"
            lines.append(f"   {connector} {child}")
            if j < len(fanout) - 1:
                lines.append("   │")

    return "\n".join(lines) if lines else None


def _build_registry_note(by_layer: Dict[str, List[str]]) -> Optional[str]:
    registry = by_layer.get("registry")
    if not registry:
        return None
    return f"{registry[0]}\n └── Container images"


def _build_cicd_flow(by_layer: Dict[str, List[str]]) -> Optional[str]:
    """cicd tool -> CI -> registry -> gitops tool -> compute hub, using
    whichever of those steps were actually named — e.g. a transcript that
    only mentions GitHub Actions (no GitOps tool) still gets a short
    "GitHub Actions -> CI" flow rather than nothing at all.
    """
    cicd = by_layer.get("cicd")
    gitops = by_layer.get("gitops")
    if not cicd and not gitops:
        return None

    nodes: List[str] = []
    if cicd:
        nodes.append(cicd[0])
        nodes.append("CI")
    if by_layer.get("registry"):
        nodes.append(by_layer["registry"][0])
    if gitops:
        nodes.append(gitops[0])
    if by_layer.get("compute"):
        nodes.append(by_layer["compute"][0])

    if len(nodes) < 2:
        return None
    return _vertical_chain(nodes)


def _build_labeled_arrow(by_layer: Dict[str, List[str]], layer: str, target_label: str) -> Optional[str]:
    values = by_layer.get(layer)
    if not values:
        return None
    return f"{values[0]} ──► {target_label}"


def _build_monitoring_flow(by_layer: Dict[str, List[str]]) -> Optional[str]:
    tools = by_layer.get("monitoring")
    if not tools:
        return None
    return f"{', '.join(tools)} ──► Observability"


def build_architecture_flow_diagram(components: List[str]) -> Optional[str]:
    """Build a top-down "mental model" diagram from a flat list of detected
    component names: the main request-flow tree, plus separate supporting-
    infrastructure flows (CI/CD pipeline, IaC, secrets, observability,
    alerting) — each shown only when actually named, never fabricated to
    fill in a "typical" architecture the transcript didn't describe.
    Returns None when nothing at all was recognized.
    """
    by_layer = _group_by_layer(components)
    if not by_layer:
        return None

    sections = [
        _build_main_flow(by_layer),
        _build_registry_note(by_layer),
        _build_cicd_flow(by_layer),
        _build_labeled_arrow(by_layer, "iac", "Infrastructure"),
        _build_labeled_arrow(by_layer, "secrets", "Secrets"),
        _build_monitoring_flow(by_layer),
        _build_labeled_arrow(by_layer, "alerting", "Alerting"),
    ]
    sections = [s for s in sections if s]
    return "\n\n".join(sections) if sections else None
