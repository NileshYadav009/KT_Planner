"""
architecture_diagram.py
========================
Builds a simple, generic top-down "mental model" diagram (Customer -> ... ->
compute -> {service, database, cache, queue}) from a flat list of detected
architecture component names (schema_generator/knowledge_builder's tools
regex output).

Deliberately NOT a general-purpose relationship extractor that parses
arbitrary transcript sentences for "X connects to Y" phrasing — that's
fragile and hard to generalize correctly across arbitrary transcripts.
Instead this uses a coarse, extensible layer classification (frontend / cdn
/ load balancer / compute / service / database / cache / queue / registry)
for known component names, and renders the common web-architecture shape
those layers imply. Only layers that were actually named in the transcript
are drawn — nothing is invented to fill in a "typical" architecture.
"""
from typing import Dict, List, Optional

# Maps a canonical component name (lowercased, as PATTERN_EXTRACTORS["tools"]
# captures it in field_populator.py) to a coarse architectural layer used for
# the flow diagram. Deliberately conservative: only layers that represent a
# genuine request-flow position are included here — supporting
# infrastructure (monitoring, alerting, CI/CD, IaC, secrets, security
# scanning) is intentionally left unmapped so it never gets force-fit into a
# request-flow diagram it was never actually part of.
_LAYER_TERMS: Dict[str, List[str]] = {
    "frontend": ["react", "angular", "vue", "vue.js"],
    "cdn": ["cloudfront"],
    "load_balancer": ["application load balancer", "alb", "load balancer"],
    "compute": ["amazon eks", "kubernetes", "docker", "rancher"],
    "service": ["fastapi", "fast api", "django", "flask", "node.js", "node", "express"],
    "database": ["postgresql", "mysql", "mongodb", "amazon rds"],
    "cache": ["redis"],
    "queue": ["kafka", "rabbitmq", "sqs", "amazon sqs"],
    "registry": ["amazon ecr"],
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


def build_architecture_flow_diagram(components: List[str]) -> Optional[str]:
    """Build a top-down tree diagram from a flat list of detected component
    names. Returns None when nothing resembling a request-flow position was
    detected (e.g. a transcript that only discussed IaC or monitoring
    tooling) — an empty or fabricated diagram would be worse than no
    diagram at all in a document meant to be trusted at face value.
    """
    by_layer: Dict[str, str] = {}
    for comp in components or []:
        layer = _classify(comp)
        if layer and layer not in by_layer:
            # First-seen name for this layer wins (e.g. if a transcript says
            # both "Amazon EKS" and "Kubernetes", whichever was detected
            # first becomes the hub's label) — components is already in
            # transcript-order from enrich_architecture_knowledge().
            by_layer[layer] = comp

    chain = [by_layer[layer] for layer in _CHAIN_LAYERS if layer in by_layer]
    fanout = [by_layer[layer] for layer in _FANOUT_LAYERS if layer in by_layer]
    registry = by_layer.get("registry")

    if not chain and not fanout and not registry:
        return None

    has_hub = "compute" in by_layer
    if not has_hub and fanout:
        # No compute/orchestration hub was actually named — nothing to
        # branch FROM, so render everything as one straight-through chain
        # instead of fabricating a branch point the transcript never stated.
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

    if registry:
        if lines:
            lines.append("")
        lines.append(registry)
        lines.append(" └── Container images")

    return "\n".join(lines) if lines else None
