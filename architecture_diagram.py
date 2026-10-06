"""
architecture_diagram.py
========================
The High-Level Architecture picture (P1-9): an SVG drawn from the components
the KT named and the connections it stated.

- Each named component is placed by what it is (component_catalog.py and
  _LAYER_TERMS below): the compute platform is a box holding the workload;
  managed data services (databases, caches, queues, storage, pipelines) sit
  beside it inside the cloud account; third-party services (Stripe, Twilio)
  sit outside the account; CI/CD and operations tools sit in lanes below.
  Before, an ASCII tree hung every database and queue off the compute node,
  which read as "runs inside the cluster".
- A line is drawn only when a sentence states the connection ("Bookings are
  written to DynamoDB", "Cloud Build deploys to Cloud Run"), labelled with
  what it does and kept with that sentence for the Connections table. A
  component named without a stated connection is drawn without a line.
- The layout is deterministic: the same KT always draws the same picture.

Only what the transcript named is drawn; nothing is added to complete a
"typical" architecture.
"""
import html
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Maps a canonical component name (lowercased, as PATTERN_EXTRACTORS["tools"]
# captures it in field_populator.py, after knowledge_builder.py's
# acronym-to-branded-name canonicalization) to a coarse architectural layer.
_LAYER_TERMS: Dict[str, List[str]] = {
    "frontend": ["react", "angular", "vue", "vue.js"],
    "dns": ["route 53", "route53", "azure dns", "cloud dns"],
    "cdn": ["cloudfront", "azure front door"],
    "waf": ["aws waf", "waf", "azure waf", "cloud armor"],
    "load_balancer": [
        "application load balancer", "alb", "load balancer", "application gateway",
        "nginx", "haproxy", "traefik", "istio",
    ],
    "compute": [
        "amazon eks", "azure kubernetes service", "aks",
        "google kubernetes engine", "gke",
        "openshift", "amazon ecs", "ecs", "fargate", "aws lambda", "cloud run",
        "azure functions", "app service", "nomad",
        "kubernetes", "docker", "rancher",
    ],
    "service": [
        "go", "python", "java",
        "fastapi", "fast api", "django", "flask", "node.js", "node", "express",
        ".net", "asp.net", "dotnet", ".net microservices", "spring boot", "spring",
    ],
    "database": [
        "postgresql", "mysql", "mongodb", "amazon rds", "azure sql",
        "aurora", "aurora postgresql", "amazon aurora", "dynamodb", "azure cosmos db",
        "bigquery", "cloud sql", "firestore", "bigtable", "cloud spanner",
        "cassandra", "cosmos db", "sql server",
    ],
    "cache": ["redis", "memorystore", "memcached"],
    "queue": [
        "kafka", "amazon msk", "msk", "rabbitmq", "sqs", "amazon sqs",
        "azure service bus", "service bus", "pub/sub",
        "activemq", "kinesis", "event hubs", "event hub", "nats",
    ],
    # Third-party APIs the workload calls out to (SMS, email, payments,
    # identity). A dependency like a database, never part of the cluster.
    "external": ["twilio", "sendgrid", "stripe", "okta", "auth0"],
    "registry": ["amazon ecr", "azure container registry", "acr", "artifact registry", "container registry"],
    "cicd": ["github actions", "gitlab ci", "jenkins", "azure devops", "cloud build", "circleci", "spinnaker", "tekton"],
    "gitops": ["argocd", "argo cd", "flux"],
    "iac": ["terraform", "bicep", "ansible", "pulumi", "cloudformation"],
    "secrets": ["vault", "azure key vault", "key vault", "consul", "secret manager", "aws kms", "kms"],
    "monitoring": [
        "prometheus", "grafana", "cloudwatch", "datadog", "splunk", "elk",
        "elasticsearch", "logstash", "kibana", "azure monitor", "application insights",
        "cloud monitoring", "cloud logging", "stackdriver", "opensearch", "amazon opensearch",
        "new relic", "sentry", "dynatrace", "loki", "jaeger",
    ],
    "alerting": ["pagerduty", "opsgenie", "victorops"],
    # Data/ML-pipeline roles that don't fit the web-request-flow chain
    # (a data platform is often not customer-facing at all) — shown as
    # their own standalone arrows instead, same as iac/secrets/alerting.
    "orchestration": ["airflow"],
    "ml_platform": ["vertex ai"],
    "data_pipeline": ["dataflow"],
    "object_storage": ["cloud storage", "cold storage", "blob storage", "azure blob storage", "s3"],
}

# Products from component_catalog.py join their layer after the entries above.
from component_catalog import layer_terms as _catalog_layer_terms

for _layer, _terms in _catalog_layer_terms().items():
    _existing = _LAYER_TERMS.setdefault(_layer, [])
    _existing.extend(t for t in _terms if t not in _existing)

# Within the "compute" layer, prefer whichever actually-mentioned term is
# most specific/branded (a managed-Kubernetes product name) over a generic
# term like bare "Kubernetes" — a transcript that says both "Amazon EKS"
# and, later, generic "Kubernetes workloads" should label the hub "Amazon
# EKS", not whichever term happened to be scanned first.
_COMPUTE_SPECIFICITY = [
    "amazon eks", "azure kubernetes service", "aks",
    "google kubernetes engine", "gke",
    "openshift", "amazon ecs", "ecs", "fargate", "aws lambda", "cloud run",
    "azure functions", "app service", "nomad",
    "kubernetes", "docker", "rancher",
]

# The main top-down request-flow chain, in order.
_CHAIN_LAYERS = ["frontend", "dns", "cdn", "waf", "load_balancer", "compute"]
# Layers that represent real evidence of an inbound, customer-facing entry
# point — as opposed to "compute", which is just a hub and, on its own,
# doesn't establish that a customer request path exists at all (e.g. a
# backend-only data-pipeline platform can run entirely on a compute hub
# with no frontend/CDN/load-balancer ever mentioned).
_ENTRY_LAYERS = ["frontend", "dns", "cdn", "waf", "load_balancer"]
# Layers that fan out FROM the compute hub, rather than continuing the chain.
# Split so the diagram can distinguish "hosted BY compute" (the app
# workloads themselves) from "DEPENDED ON by the workloads" (a database,
# cache, or queue) — rendering both as identical tree children previously
# implied every one of them was hosted inside the compute node/cluster,
# which is only true for "service".
_HOSTED_LAYERS = ["service"]
_DEPENDENCY_LAYERS = ["database", "cache", "queue", "external"]
_FANOUT_LAYERS = _HOSTED_LAYERS + _DEPENDENCY_LAYERS

_ROOT_LABEL = "Customer"

# A managed service and the engine it provides ("Kafka is provided through
# Amazon MSK") are one dependency, not two. Drawn as "Kafka (Amazon MSK)".
_MANAGED_OFFERING_OF = {
    "amazon msk": "kafka",
    "msk": "kafka",
    "aurora postgresql": "postgresql",
    "amazon aurora": "postgresql",
    "azure cache for redis": "redis",
    "memorystore": "redis",
}


# A bare database engine named alongside a managed database service is that
# service's engine ("Amazon RDS with PostgreSQL"), not a second database.
_DB_ENGINES = {"postgresql", "mysql", "postgres"}
_MANAGED_DATABASES = {"amazon rds", "azure sql", "cloud sql", "aurora", "amazon aurora", "aurora postgresql"}


def _merge_managed_offerings(components: List[str]) -> List[str]:
    if any(c.lower() in _MANAGED_DATABASES for c in components):
        components = [c for c in components if c.lower() not in _DB_ENGINES]
    lowered = {c.lower(): c for c in components}
    merged: List[str] = []
    consumed = set()
    for comp in components:
        low = comp.lower()
        if low in consumed:
            continue
        engine = _MANAGED_OFFERING_OF.get(low)
        if engine and engine in lowered:
            # Rendered at the engine's position, below.
            continue
        offering = next(
            (lowered[o] for o, e in _MANAGED_OFFERING_OF.items() if e == low and o in lowered),
            None,
        )
        if offering:
            merged.append(f"{comp} ({offering})")
            consumed.add(offering.lower())
        else:
            merged.append(comp)
    return merged


def _classify(component: str) -> Optional[str]:
    lowered = component.strip().lower()
    for layer, names in _LAYER_TERMS.items():
        if lowered in names:
            return layer
    return None


def _reorder_compute_by_specificity(members: List[str]) -> List[str]:
    def rank(term: str) -> int:
        lowered = term.strip().lower()
        try:
            return _COMPUTE_SPECIFICITY.index(lowered)
        except ValueError:
            return len(_COMPUTE_SPECIFICITY)
    return sorted(members, key=rank)


def _group_by_layer(components: List[str]) -> Dict[str, List[str]]:
    """Every detected component, grouped by layer, preserving the
    transcript-order each component was first seen in (components is
    already deduplicated by the time it reaches here) — except "compute",
    which is reordered by specificity (see _COMPUTE_SPECIFICITY)."""
    grouped: Dict[str, List[str]] = defaultdict(list)
    for comp in components or []:
        layer = _classify(comp)
        if layer:
            grouped[layer].append(comp)
    if "compute" in grouped:
        grouped["compute"] = _reorder_compute_by_specificity(grouped["compute"])
    return grouped

# --------------------------------------------------------------------------
# Components -> graph
# --------------------------------------------------------------------------

_CAPTIONS = {
    "frontend": "frontend", "dns": "DNS", "cdn": "CDN", "waf": "firewall", "load_balancer": "load balancer",
    "database": "database", "cache": "cache", "queue": "messaging", "object_storage": "object storage",
    "orchestration": "workflow orchestration", "ml_platform": "ML platform", "data_pipeline": "data pipeline",
    "external": "third-party service", "cicd": "CI/CD", "registry": "image registry", "gitops": "GitOps",
    "iac": "infrastructure as code", "monitoring": "monitoring", "alerting": "alerting", "secrets": "secrets",
}
# Managed data and processing services: inside the cloud account, outside
# the compute platform.
_DATA_LAYERS = ["database", "cache", "queue", "object_storage", "data_pipeline", "orchestration", "ml_platform"]
_DELIVERY_LAYERS = ["cicd", "registry", "gitops"]
_OPERATIONS_LAYERS = ["monitoring", "alerting", "secrets"]
# Components that can be the subject of a stated data flow ("A Dataflow job
# writes them to BigQuery"); a database or a queue is acted on, not acting.
_ACTOR_LAYERS = {"data_pipeline", "orchestration", "ml_platform", "frontend"}
_GENERIC_COMPUTE = {"kubernetes", "docker", "rancher"}

_VENDORS = [
    ("AWS", re.compile(r"^(?:amazon|aws)\b|\b(?:dynamodb|cloudfront|route ?53|fargate|ecs|eks|cloudformation|cloudwatch|"
                       r"aurora|sqs|sns|kinesis|msk|s3|elasticache|lambda|ec2|ecr)\b", re.IGNORECASE)),
    ("Microsoft Azure", re.compile(r"\bazure\b|\b(?:aks|cosmos db|bicep|application insights|key vault|"
                                   r"service bus|event hubs?|app service)\b", re.IGNORECASE)),
    ("Google Cloud", re.compile(r"^google\b|\b(?:gke|cloud run|bigquery|pub/sub|cloud sql|firestore|bigtable|"
                                r"spanner|dataflow|vertex ai|memorystore|cloud storage|cloud monitoring|cloud logging|"
                                r"cloud build|artifact registry|cloud armor|cloud dns|cloud functions)\b", re.IGNORECASE)),
]

# What a sentence says one component does to another. Matched only in the
# words between the source and the target (or the target's own clause).
_WRITE_RE = re.compile(r"\b(?:writ(?:e|es|ten|ing)|stor(?:e|es|ed|ing)|persist\w*|sav(?:e|es|ed)|insert\w*|"
                       r"archiv\w*|upload\w*|push(?:es|ed)?|kept|land(?:s|ed)?|go(?:es)? (?:in)?to)\b", re.IGNORECASE)
_READ_RE = re.compile(r"\b(?:read(?:s|ing)?|quer(?:y|ies|ied)|fetch\w*|look(?:s|ed)? up|load(?:s|ed)? from)\b",
                      re.IGNORECASE)
_CACHE_RE = re.compile(r"\b(?:cach\w*|session\w*)\b", re.IGNORECASE)
_PUBLISH_RE = re.compile(r"\b(?:publish\w*|emit\w*|produc\w*|enqueue\w*|queue[sd]?|send(?:s|ing)? (?:messages?|events?|"
                         r"jobs?)|events? (?:go|are sent) to)\b|\bto queue\b", re.IGNORECASE)
_CONSUME_RE = re.compile(r"\b(?:consum\w*|subscrib\w*|listen\w*|pull\w*|process(?:es|ed)? (?:messages|events|jobs) "
                         r"from)\b", re.IGNORECASE)
_CALL_RE = re.compile(r"\b(?:call\w*|go(?:es)? through|via|sent (?:with|through|via)|send\w* (?:with|through|via)|"
                      r"integrat\w*|talk\w* to|hit\w*|handled by|powered by|process(?:es|ed)? (?:by|with|through))\b",
                      re.IGNORECASE)
_USE_RE = re.compile(r"\b(?:use[sd]?|using|backed by|on top of|relies on|depends on|connects? to|runs? on)\b",
                     re.IGNORECASE)
_LOCATED_RE = re.compile(r"\b(?:is|are|lives?|sits?)\s+(?:in|on)\b", re.IGNORECASE)
# "Redis is used for caching", "PostgreSQL is the primary database",
# "Service Bus to queue invoice jobs".
_DEFINED_AS_RE = re.compile(r"^\s*(?:(?:is|are)\s+(?:used\s+)?(?:for|as|our|the)|to|for|as)\b", re.IGNORECASE)
_DEPLOY_RE = re.compile(r"\b(?:deploy\w*|release\w*|ship\w*|roll\w* out|sync\w*|promot\w*)\b", re.IGNORECASE)
_MONITOR_RE = re.compile(r"\b(?:monitor\w*|watch\w*|dashboard\w*|metric\w*|logs?|logging|observ\w*|trac\w*|alert\w*)\b",
                         re.IGNORECASE)
_SECRETS_RE = re.compile(r"\b(?:secret\w*|credential\w*|certificat\w*|keys?)\b", re.IGNORECASE)
_PROVISION_RE = re.compile(r"\b(?:defined|provision\w*|managed|built|created|described|codified|kept|lives?|is|are)\s+"
                           r"(?:in|with|by|using|through)\b|\bas code\b|\binfrastructure\b", re.IGNORECASE)
# Who did it, when the subject is not a component: the system itself
# ("we", "the API", the system's name) or a passive / data-moving phrase
# ("Bookings are written to", "payments go through"). "Devices publish to
# Pub/Sub" names another actor, so no line is drawn from the system.
_SYSTEM_SUBJECT_RE = re.compile(
    r"^\W*(?:(?:and|then|so|also|which|that)\s+)*(?:we|it|they|our\s+\w+|the\s+(?:\w+\s+)?(?:app|application|api|"
    r"service|backend|system|platform|workload|worker|pods?|code)|this\s+\w+)\b", re.IGNORECASE)
_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|gets?|got|be|been|being)\s+(?:\w+ly\s+)?(?:\w+ed|written|sent|kept|held|run|built|made|put|"
    r"stored|done)\b|\b(?:go|goes|flow|flows|land|lands|live|lives|sit|sits)\b|\b(?:is|are)\s+(?:in|on)\b", re.IGNORECASE)
_CLAUSE_BREAK_RE = re.compile(r"[;:]|,\s*(?:and|but|while|so)\b|\.\s", re.IGNORECASE)


def _vendor(name: str) -> Optional[str]:
    for vendor, pattern in _VENDORS:
        if pattern.search(name):
            return vendor
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "node"


def _merge_with_aliases(components: List[str]) -> List[Tuple[str, List[str]]]:
    """(label, member names) per drawn node: a managed offering and its
    engine are one node ("Kafka (Amazon MSK)")."""
    merged = _merge_managed_offerings(components)
    lowered = {c.lower(): c for c in components}
    out = []
    for label in merged:
        members = [label]
        inner = re.match(r"^(.*) \((.*)\)$", label)
        if inner:
            members = [inner.group(1), inner.group(2)]
        out.append((label, [m for m in members if m.lower() in lowered] or [label]))
    return out


def _clause_of(sentence: str, start: int, end: int) -> str:
    """The clause around a mention: from the previous clause break to the next."""
    before = [m.end() for m in _CLAUSE_BREAK_RE.finditer(sentence, 0, start)]
    after = _CLAUSE_BREAK_RE.search(sentence, end)
    return sentence[(before[-1] if before else 0):(after.start() if after else len(sentence))]


def _label_for(layer: str, window: str, defining: str) -> Optional[str]:
    """The stated relationship to a component of `layer`, from the words
    leading up to it (`window`) or right after it (`defining`)."""
    text = window + " " + defining
    if layer == "database":
        if _WRITE_RE.search(window):
            return "writes to"
        if _READ_RE.search(window):
            return "reads from"
        if _USE_RE.search(text) or _LOCATED_RE.search(window) or defining:
            return "stores data in"
    elif layer == "object_storage":
        if re.search(r"\barchiv", window, re.IGNORECASE):
            return "archives to"
        if _WRITE_RE.search(window):
            return "writes to"
        if _READ_RE.search(window):
            return "reads from"
        if _USE_RE.search(text) or _LOCATED_RE.search(window) or defining:
            return "stores files in"
    elif layer == "cache":
        if _READ_RE.search(window):
            return "reads from"
        if _CACHE_RE.search(text) or _USE_RE.search(text) or defining:
            return "caches in"
    elif layer == "queue":
        if _CONSUME_RE.search(text):
            return "consumes from"
        if _PUBLISH_RE.search(text):
            return "publishes to"
        if _USE_RE.search(text) or defining:
            return "sends messages via"
    elif layer == "external":
        if _CALL_RE.search(window) or _USE_RE.search(window) or _PASSIVE_RE.search(window):
            return "calls"
    elif layer in ("data_pipeline", "orchestration", "ml_platform"):
        if _USE_RE.search(text) or defining:
            return "uses"
    return None


def _mentions(sentence: str, aliases: Dict[str, List[str]]) -> List[Tuple[int, int, str]]:
    """(start, end, node id) for every node named in the sentence, longest
    spelling first, no overlaps."""
    found: List[Tuple[int, int, str]] = []
    pairs = sorted(((a, nid) for nid, names in aliases.items() for a in names), key=lambda p: -len(p[0]))
    taken: List[Tuple[int, int]] = []
    for alias, nid in pairs:
        for m in re.finditer(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", sentence, re.IGNORECASE):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.end(), nid))
    return sorted(found)


def build_architecture_graph(
    components: List[str],
    sentences: Optional[List[str]] = None,
    aliases: Optional[Dict[str, List[str]]] = None,
    system_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The architecture as boxes and stated connections.

    Placement follows what each component is: the compute platform is a
    box holding the workload; managed data services sit beside it inside
    the cloud account; third-party services sit outside the account;
    delivery and operations tools sit in their own lanes. A connection is
    drawn only when a sentence states it ("Bookings are written to
    DynamoDB"), labelled with what it does (writes to, reads from, calls,
    publishes to, deploys to, monitors) and kept with that sentence.
    Returns None when fewer than two components were named.
    """
    by_layer = _group_by_layer(components)
    sentences = [s.strip() for s in (sentences or []) if s and s.strip()]
    raw_aliases = {k.lower(): {a.lower() for a in v} for k, v in (aliases or {}).items()}
    node_layer: Dict[str, str] = {}
    node_aliases: Dict[str, List[str]] = {}

    def node(label: str, layer: str, members: Optional[List[str]] = None) -> Dict[str, str]:
        nid = _slug(label)
        names = set()
        for member in members or [label]:
            names.add(member.lower())
            names |= raw_aliases.get(member.lower(), set())
        node_layer[nid] = layer
        node_aliases[nid] = sorted(names, key=len, reverse=True)
        # Named only in lowercase in the transcript ("application load balancer").
        shown = label if any(ch.isupper() for ch in label) else label.title()
        return {"id": nid, "label": shown, "caption": _CAPTIONS.get(layer, layer.replace("_", " "))}

    compute = by_layer.get("compute", [])
    specific = [c for c in compute if c.lower() not in _GENERIC_COMPUTE]
    compute_names = specific or compute[:1]
    services = by_layer.get("service", [])

    entry = [node(by_layer[layer][0], layer) for layer in _CHAIN_LAYERS if layer != "compute" and layer in by_layer]
    if entry:
        entry.insert(0, {"id": "users", "label": "Users", "caption": "customers"})
    data = [node(label, layer, members) for layer in _DATA_LAYERS
            for label, members in _merge_with_aliases(by_layer.get(layer, []))]
    external = [node(c, "external") for c in by_layer.get("external", [])]
    delivery = [node(c, layer) for layer in _DELIVERY_LAYERS for c in by_layer.get(layer, [])[:2]]
    operations = [node(c, layer) for layer in _OPERATIONS_LAYERS for c in by_layer.get(layer, [])[:2]]
    iac = by_layer.get("iac", [])

    drawn = len(entry) + len(data) + len(external) + len(delivery) + len(operations) + len(iac) \
        + (1 if compute_names or services else 0)
    if drawn < 2:
        return None

    # The workload: the system being handed over, on its compute platform.
    for name in list(compute) + list(services):
        node_aliases.setdefault("central", [])
        node_aliases["central"] = sorted(set(node_aliases["central"]) | {name.lower()} | raw_aliases.get(name.lower(), set()),
                                         key=len, reverse=True)
    if system_name:
        node_aliases.setdefault("central", [])
        node_aliases["central"] = sorted(set(node_aliases["central"]) | {system_name.lower()}, key=len, reverse=True)
    node_layer["central"] = "compute"
    # With no platform, workload, data store, third party or entry point
    # named (only CI/CD or monitoring tools), there is nothing to put in the
    # middle: only the lanes are drawn.
    has_core = bool(compute or services or data or external or entry)
    central = {
        "id": "central",
        "header": " · ".join(compute_names) if compute_names else None,
        "label": system_name or "Application",
        "caption": " · ".join(services) if services else "application",
    } if has_core else None

    # Cloud account boundary: named when every vendor-specific component
    # belongs to one cloud.
    vendors = {v for v in (_vendor(c) for c in components) if v}
    boundary = None
    if has_core and (compute_names or data or vendors):
        boundary = {"label": vendors.pop() if len(vendors) == 1 else ("Cloud platform" if vendors else "Platform"),
                    "subtitle": None}

    edges: List[Dict[str, str]] = []
    seen_edges = set()

    def add_edge(src: str, dst: str, label: str, quote: str, kind: str) -> None:
        if (src, dst) in seen_edges or src == dst:
            return
        seen_edges.add((src, dst))
        edges.append({"from": src, "to": dst, "label": label, "quote": quote, "kind": kind})

    system_word = re.escape(system_name.lower()) if system_name else None
    targets = {n["id"] for n in data + external}
    for sentence in sentences:
        spots = _mentions(sentence, node_aliases)
        for idx, (start, end, nid) in enumerate(spots):
            if nid not in targets:
                continue
            layer = node_layer[nid]
            after = spots[idx + 1][0] if idx + 1 < len(spots) else len(sentence)
            brk = _CLAUSE_BREAK_RE.search(sentence, end, after)
            tail = sentence[end:(brk.start() if brk else after)]
            defining = tail if _DEFINED_AS_RE.match(tail) else ""
            source, window = None, None
            for pstart, pend, pid in reversed(spots[:idx]):
                segment = sentence[pend:start]
                if pid == "central":
                    source, window = "central", segment
                    break
                if node_layer.get(pid) in _ACTOR_LAYERS and not _CLAUSE_BREAK_RE.search(segment) \
                        and _label_for(layer, segment, ""):
                    source, window = pid, segment
                    break
            if source is None:
                clause = _clause_of(sentence, start, end)
                window = clause[:clause.lower().find(sentence[start:end].lower())] if clause else sentence[:start]
                subject_ok = (_SYSTEM_SUBJECT_RE.match(window) or _PASSIVE_RE.search(window)
                              or (system_word and re.search(system_word, window.lower())) or defining)
                if not subject_ok:
                    continue
                source = "central"
            label = _label_for(layer, window, defining)
            if label:
                add_edge(source, nid, label, sentence, "data" if layer != "external" else "external")

    # Delivery and operations: stated when the tool is named in a sentence
    # about that job. The quote is the sentence where the tool and the job
    # are mentioned closest together ("Secrets live in Key Vault", not a
    # failure story that mentions a Key Vault secret in passing). Which
    # alerting tool a monitoring tool pages through is not inferred from
    # both being named in one sentence; the alerting tool is shown without
    # a line.
    jobs = {"cicd": (_DEPLOY_RE, "deploys to", "delivery"), "gitops": (_DEPLOY_RE, "deploys to", "delivery"),
            "monitoring": (_MONITOR_RE, "monitors", "operations"),
            "secrets": (_SECRETS_RE, "supplies secrets to", "operations")}
    best: Dict[str, Tuple[int, str]] = {}
    for sentence in sentences:
        for start, end, nid in _mentions(sentence, node_aliases):
            job = jobs.get(node_layer.get(nid))
            if not job:
                continue
            cues = [m for m in job[0].finditer(sentence)]
            if cues:
                gap = min(min(abs(m.start() - end), abs(start - m.end())) for m in cues)
                if nid not in best or gap < best[nid][0]:
                    best[nid] = (gap, sentence)
    for nid, (_, sentence) in best.items():
        pattern, label, kind = jobs[node_layer[nid]]
        if central is not None:
            add_edge(nid, "central", label, sentence, kind)

    for sentence in sentences:
        if iac and boundary and not boundary["subtitle"]:
            for name in iac:
                spellings = {name.lower()} | raw_aliases.get(name.lower(), set())
                if any(re.search(r"(?<![\w-])" + re.escape(s) + r"(?![\w-])", sentence, re.IGNORECASE)
                       for s in spellings) and _PROVISION_RE.search(sentence):
                    boundary["subtitle"] = f"Provisioned with {name}"
                    boundary["quote"] = sentence
                    break
    if iac and not (boundary and boundary.get("subtitle")):
        delivery.extend(node(name, "iac") for name in iac)

    # A component that writes to another (Dataflow -> BigQuery) is listed
    # just above its targets, so the line between them stays short.
    feeds: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        if e["from"] != "central" and e["kind"] == "data":
            feeds[e["from"]].append(e["to"])
    fed = {t for outs in feeds.values() for t in outs}
    by_id = {n["id"]: n for n in data}
    ordered = [n for n in data if n["id"] not in feeds and n["id"] not in fed]
    for actor, outs in feeds.items():
        ordered.append(by_id[actor])
        ordered.extend(by_id[t] for t in outs if by_id[t] not in ordered)
    data = ordered + [n for n in data if n not in ordered]

    return {"boundary": boundary, "central": central, "entry": entry, "data": data, "external": external,
            "delivery": delivery, "operations": operations, "edges": edges}


def describe_connections(graph: Dict[str, Any]) -> List[Dict[str, str]]:
    """The stated connections as table rows, each with the KT sentence."""
    names = {n["id"]: n["label"] for key in ("entry", "data", "external", "delivery", "operations") for n in graph[key]}
    central = graph["central"] or {"label": "the system"}
    names["central"] = central["label"] + (f" ({central['header']})" if central.get("header") else "")
    rows = [{"From": names.get(e["from"], e["from"]), "Connection": e["label"], "To": names.get(e["to"], e["to"]),
             "Said in the KT": e["quote"]} for e in graph["edges"]]
    boundary = graph.get("boundary") or {}
    if boundary.get("subtitle"):
        rows.append({"From": boundary["subtitle"].replace("Provisioned with ", ""), "Connection": "provisions",
                     "To": boundary["label"], "Said in the KT": boundary.get("quote", "")})
    return rows


# --------------------------------------------------------------------------
# Graph -> SVG
# --------------------------------------------------------------------------

_FONT = "Helvetica, Arial, sans-serif"
_STYLE = {
    "entry": ("#f1f5f9", "#475569"),
    "data": ("#f0fdfa", "#0f766e"),
    "external": ("#fff7ed", "#c2410c"),
    "delivery": ("#f8fafc", "#64748b"),
    "operations": ("#fefce8", "#a16207"),
    "workload": ("#ffffff", "#6366f1"),
}
_CHAR_W = 6.3        # average glyph width at 11.5px
_LINE_H = 14
_NODE_GAP = 18


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _wrap(text: str, width: float) -> List[str]:
    limit = max(6, int((width - 14) / _CHAR_W))
    lines, line = [], ""
    for word in str(text).split():
        if line and len(line) + 1 + len(word) > limit:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines or [""]


def _node_height(n: Dict[str, str], width: float) -> float:
    return 14 + _LINE_H * len(_wrap(n["label"], width)) + 13


class _Canvas:
    def __init__(self):
        self.parts: List[str] = []

    def rect(self, x, y, w, h, fill, stroke, rx=7, dashed=False, width=1.2):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{width}"{dash}/>')

    def text(self, x, y, text, size=11.5, weight="normal", fill="#0f172a", anchor="middle"):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
                          f'text-anchor="{anchor}">{_esc(text)}</text>')

    def node(self, x, y, w, n, style):
        fill, stroke = _STYLE[style]
        lines = _wrap(n["label"], w)
        h = _node_height(n, w)
        self.rect(x, y, w, h, fill, stroke)
        for i, line in enumerate(lines):
            self.text(x + w / 2, y + 18 + i * _LINE_H, line, weight="bold" if style == "workload" else "normal")
        self.text(x + w / 2, y + 18 + len(lines) * _LINE_H, n.get("caption", ""), size=9.5, fill="#64748b")
        return h

    def path(self, points, dashed=False):
        d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(points))
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="#475569" stroke-width="1.3"{dash} '
                          f'marker-end="url(#arrow)"/>')

    def label(self, x, y, text, anchor="middle"):
        self.text(x, y, text, size=9.5, fill="#334155", anchor=anchor)


def render_architecture_svg(graph: Dict[str, Any]) -> str:
    """A self-contained SVG (no scripts, no external references), laid out
    deterministically so the same KT always draws the same picture."""
    has_external = bool(graph["external"])
    W = 760 if has_external else 540
    c = _Canvas()
    y = 14

    # Request path: Users -> CDN/WAF/load balancer -> the workload.
    entry_bottom, entry_last_center = None, None
    if graph["entry"]:
        gap = 30
        ew = min(112, (W - 24 - (len(graph["entry"]) - 1) * gap) / len(graph["entry"]))
        total = len(graph["entry"]) * ew + (len(graph["entry"]) - 1) * gap
        x = max(12, (W - total) / 2)
        heights = [_node_height(n, ew) for n in graph["entry"]]
        row_h = max(heights)
        for i, n in enumerate(graph["entry"]):
            c.node(x, y, ew, n, "entry")
            if i:
                c.path([(x - gap, y + heights[i - 1] / 2), (x - 2, y + heights[i - 1] / 2)])
            entry_last_center = x + ew / 2
            x += ew + gap
        entry_bottom = y + row_h
        y = entry_bottom + 30

    if graph["central"] is None:
        # Only delivery or operations tools were named: just their lanes.
        main_bottom = y - 48
        box_x = box_y = box_w = box_h = 0
    else:
        # The cloud account, the compute platform and the managed services.
        boundary = graph.get("boundary")
        band_top = y
        inner_top = band_top + (44 if boundary and boundary.get("subtitle") else 34) if boundary else band_top
        data_x, data_w = 40, 168
        box_x, box_w = 300, 220
        ext_x, ext_w = 610, 138

        data_pos = {}
        yy = inner_top
        for n in graph["data"]:
            h = _node_height(n, data_w)
            data_pos[n["id"]] = (yy, h)
            yy += h + _NODE_GAP
        data_bottom = yy - _NODE_GAP if graph["data"] else inner_top

        ext_pos = {}
        yy = inner_top
        for n in graph["external"]:
            h = _node_height(n, ext_w)
            ext_pos[n["id"]] = (yy, h)
            yy += h + _NODE_GAP
        ext_bottom = yy - _NODE_GAP if graph["external"] else inner_top

        central = graph["central"]
        header_lines = _wrap(central["header"], box_w - 16) if central.get("header") else []
        workload = {"label": central["label"], "caption": central["caption"]}
        work_w = box_w - 36
        work_h = _node_height(workload, work_w)
        header_h = (14 + _LINE_H * len(header_lines) + 14) if header_lines else 8
        box_h = max(header_h + work_h + 18, data_bottom - inner_top, ext_bottom - inner_top, 96)
        box_y = inner_top
        if boundary:
            b_right = 534 if has_external else W - 12
            b_bottom = max(box_y + box_h, data_bottom) + 18
            c.rect(12, band_top, b_right - 12, b_bottom - band_top, "#f8fafc", "#94a3b8", rx=10, dashed=True)
            c.text(24, band_top + 19, boundary["label"], size=11.5, weight="bold", fill="#334155", anchor="start")
            if boundary.get("subtitle"):
                c.text(24, band_top + 33, boundary["subtitle"], size=9.5, fill="#64748b", anchor="start")
            main_bottom = max(b_bottom, ext_bottom)
        else:
            b_bottom = max(box_y + box_h, data_bottom)
            main_bottom = max(b_bottom, ext_bottom)

        if central.get("header"):
            c.rect(box_x, box_y, box_w, box_h, "#eef2ff", "#4f46e5", rx=9)
            for i, line in enumerate(header_lines):
                c.text(box_x + box_w / 2, box_y + 18 + i * _LINE_H, line, weight="bold", fill="#3730a3")
            c.text(box_x + box_w / 2, box_y + 18 + len(header_lines) * _LINE_H, "compute platform", size=9.5, fill="#6366f1")
            c.node(box_x + 18, box_y + header_h, work_w, workload, "workload")
        else:
            # No platform named: the workload itself, as tall as the columns
            # beside it so every line meets it.
            fill, stroke = _STYLE["workload"]
            c.rect(box_x, box_y, box_w, box_h, fill, stroke)
            lines = _wrap(workload["label"], box_w)
            for i, line in enumerate(lines):
                c.text(box_x + box_w / 2, box_y + 18 + i * _LINE_H, line, weight="bold")
            c.text(box_x + box_w / 2, box_y + 18 + len(lines) * _LINE_H, workload["caption"], size=9.5, fill="#64748b")

        for n in graph["data"]:
            ny, nh = data_pos[n["id"]]
            c.node(data_x, ny, data_w, n, "data")
        for n in graph["external"]:
            ny, nh = ext_pos[n["id"]]
            c.node(ext_x, ny, ext_w, n, "external")

        if entry_bottom is not None:
            top = box_y
            c.path([(entry_last_center, entry_bottom), (entry_last_center, (entry_bottom + top) / 2),
                    (box_x + box_w / 2, (entry_bottom + top) / 2), (box_x + box_w / 2, top - 2)])

        def clamp(v):
            return min(max(v, box_y + 10), box_y + box_h - 10)

        # Stated data flows. From another data component: drawn down the
        # data column's left side; from the workload: straight across.
        side_lane = 0
        for e in graph["edges"]:
            if e["kind"] == "data":
                ty, th = data_pos[e["to"]]
                tcy = ty + th / 2
                if e["from"] == "central":
                    c.path([(box_x, clamp(tcy)), (data_x + data_w + 2, clamp(tcy))])
                    c.label((box_x + data_x + data_w) / 2, clamp(tcy) - 5, e["label"])
                elif e["from"] in data_pos:
                    fy, fh = data_pos[e["from"]]
                    lane_x = data_x - 12 - side_lane * 6
                    side_lane = (side_lane + 1) % 2
                    c.path([(data_x, fy + fh / 2), (lane_x, fy + fh / 2), (lane_x, tcy), (data_x - 2, tcy)])
                    c.label(data_x + 6, fy + fh + 12, e["label"], anchor="start")
            elif e["kind"] == "external":
                ty, th = ext_pos[e["to"]]
                tcy = clamp(ty + th / 2)
                c.path([(box_x + box_w, tcy), (ext_x - 2, tcy)])
                c.label((box_x + box_w + ext_x) / 2 + 8, tcy - 5, e["label"])

    # Delivery and operations lanes below, joined to the workload by the
    # stated "deploys to" / "monitors" lines.
    lanes = [("Delivery", graph["delivery"], "delivery"), ("Operations", graph["operations"], "operations")]
    lanes = [lane for lane in lanes if lane[1]]
    bottom = main_bottom
    if lanes:
        lane_top = main_bottom + 48
        channel = main_bottom + 24
        half = (W - 36) / 2
        heights = []
        for i, (title, nodes, style) in enumerate(lanes):
            lx = 12 + i * (half + 12) if len(lanes) == 2 else 12
            lw = half if len(lanes) == 2 else W - 24
            per_row = max(1, int((lw - 16) // 118))
            nw = min(130, (lw - 16 - (per_row - 1) * 10) / per_row)
            rows = [nodes[k:k + per_row] for k in range(0, len(nodes), per_row)]
            yy = lane_top + 26
            for row in rows:
                rh = max(_node_height(n, nw) for n in row)
                for j, n in enumerate(row):
                    c.node(lx + 10 + j * (nw + 10), yy, nw, n, style)
                yy += rh + 10
            lh = yy - lane_top
            c.parts.insert(0, f'<rect x="{lx:.1f}" y="{lane_top:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="9" '
                              f'fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
            c.text(lx + 12, lane_top + 17, title, size=10.5, weight="bold", fill="#475569", anchor="start")
            heights.append(lane_top + lh)
            stated = sorted({e["label"] for e in graph["edges"] if e["kind"] == style and e["to"] == "central"})
            if stated:
                start_x = lx + lw / 2
                target_x = box_x + (50 if style == "delivery" else box_w - 50)
                target_y = box_y + box_h
                c.path([(start_x, lane_top), (start_x, channel), (target_x, channel), (target_x, target_y + 2)],
                       dashed=True)
                # Delivery's label to the right of its line, operations' to the
                # left, so neither runs off the canvas.
                if style == "delivery":
                    c.label(target_x + 6, (channel + target_y) / 2 + 3, " · ".join(stated), anchor="start")
                else:
                    c.label(target_x - 6, (channel + target_y) / 2 + 3, " · ".join(stated), anchor="end")
        bottom = max(heights)

    height = bottom + 14
    defs = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height:.0f}" width="{W}" '
            f'height="{height:.0f}" font-family="{_FONT}">{defs}' + "".join(c.parts) + "</svg>")
