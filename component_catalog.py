"""
component_catalog.py
====================
One place to teach Continuum a technology product: how it is spelled, how it
is displayed, which Technology Summary category it belongs to and where it is
drawn in the architecture diagram. The tool recogniser
(field_populator.PATTERN_EXTRACTORS["tools"]), the diagram layers
(architecture_diagram._LAYER_TERMS), the Technology Summary categories
(renderers/sections/system_overview._TECH_CATEGORY_MAP) and display-name
canonicalization (knowledge_builder._canonicalize_component_term) all read
this catalog in addition to their own original hand-written entries.

To add a product, add one Product(...) line below. Nothing else changes.

Rules for an entry:
- `spellings` are literal forms as spoken or written. Spaces and hyphens are
  interchangeable when matching ("cert manager" == "cert-manager").
- `case_sensitive=True` for names that are also ordinary English words
  (Harbor, Temporal, Backstage, Chef, Puppet...): their single-word
  spellings only count when capitalized, like the existing Node/React/Spring
  rule. Multi-word spellings ("Amazon Bedrock") match in any case, so do not
  list a multi-word spelling that is also an ordinary phrase.
- `layer` is an architecture_diagram layer, or None for tools that are not
  drawn (security scanners, cost tools, developer portals). Only "compute"
  products become the diagram's hub, so do not put a scaler or an
  autoscaler there.
- Concepts (cluster, namespace, monitoring) are not products and never
  belong here.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Product:
    name: str
    category: str
    layer: Optional[str]
    spellings: Tuple[str, ...] = field(default_factory=tuple)
    case_sensitive: bool = False


P = Product

CATALOG: List[Product] = [
    # --- AWS
    P("Amazon EC2", "Compute", "compute", ("Amazon EC2", "AWS EC2", "EC2")),
    P("AWS Lambda", "Compute", "compute", ("Lambda",), case_sensitive=True),
    P("AWS Elastic Beanstalk", "Compute", "compute", ("Elastic Beanstalk",)),
    P("Amazon ElastiCache", "Cache", "cache", ("Amazon ElastiCache", "ElastiCache", "Elasticache")),
    P("Amazon SNS", "Messaging", "queue", ("Amazon SNS", "AWS SNS", "SNS")),
    P("Amazon EventBridge", "Messaging", "queue", ("Amazon EventBridge", "EventBridge", "Event Bridge")),
    P("AWS Step Functions", "Workflow orchestration", "orchestration", ("AWS Step Functions", "Step Functions")),
    P("AWS Secrets Manager", "Secrets", "secrets", ("AWS Secrets Manager", "Secrets Manager")),
    P("AWS CloudTrail", "Security", None, ("AWS CloudTrail", "CloudTrail", "Cloud Trail")),
    P("AWS IAM", "Security", None, ("AWS IAM", "IAM")),
    P("Amazon VPC", "Networking", None, ("Amazon VPC", "AWS VPC", "VPC")),
    P("Amazon Redshift", "Database", "database", ("Amazon Redshift", "Redshift")),
    P("Amazon Athena", "Data processing", "data_pipeline", ("Amazon Athena", "Athena"), case_sensitive=True),
    P("Amazon EBS", "Storage", None, ("Amazon EBS", "EBS")),
    P("Amazon EFS", "Storage", None, ("Amazon EFS", "EFS")),
    P("Amazon S3 Glacier", "Storage", "object_storage", ("S3 Glacier", "Glacier"), case_sensitive=True),
    P("AWS CodePipeline", "GitOps / deployment", "cicd", ("AWS CodePipeline", "CodePipeline", "Code Pipeline")),
    P("AWS CodeBuild", "GitOps / deployment", "cicd", ("AWS CodeBuild", "CodeBuild")),
    P("AWS CodeDeploy", "GitOps / deployment", "cicd", ("AWS CodeDeploy", "CodeDeploy")),
    P("Amazon SageMaker", "Machine learning", "ml_platform", ("Amazon SageMaker", "SageMaker", "Sage Maker")),
    P("Amazon Bedrock", "Machine learning", "ml_platform", ("Amazon Bedrock", "AWS Bedrock", "Bedrock"), case_sensitive=True),
    # --- Azure / GCP
    P("Microsoft Entra ID", "Security", None, ("Microsoft Entra ID", "Entra ID", "Azure Active Directory", "Azure AD")),
    P("Azure Pipelines", "GitOps / deployment", "cicd", ("Azure Pipelines",)),
    P("ARM templates", "Infrastructure", "iac", ("ARM templates", "ARM template"), case_sensitive=True),
    P("Azure VNet", "Networking", None, ("Azure VNet", "VNet")),
    P("Cloud Functions", "Compute", "compute", ("Google Cloud Functions", "Cloud Functions")),
    # --- Source control (delivery tooling)
    P("GitHub", "Source control", None, ("GitHub",)),
    P("GitLab", "Source control", None, ("GitLab",)),
    P("Bitbucket", "Source control", None, ("Bitbucket",)),
    # --- CI/CD and delivery
    P("Travis CI", "GitOps / deployment", "cicd", ("Travis CI",)),
    P("TeamCity", "GitOps / deployment", "cicd", ("TeamCity", "Team City")),
    P("Bamboo", "GitOps / deployment", "cicd", ("Bamboo",), case_sensitive=True),
    P("Drone CI", "GitOps / deployment", "cicd", ("Drone CI",)),
    P("Argo Rollouts", "GitOps / deployment", None, ("Argo Rollouts",)),
    P("Argo Workflows", "Workflow orchestration", "orchestration", ("Argo Workflows",)),
    P("Flagger", "GitOps / deployment", None, ("Flagger",)),
    # --- Infrastructure as code
    P("OpenTofu", "Infrastructure", "iac", ("OpenTofu", "Open Tofu")),
    P("Terragrunt", "Infrastructure", "iac", ("Terragrunt",)),
    P("Crossplane", "Infrastructure", "iac", ("Crossplane",)),
    P("Chef", "Infrastructure", "iac", ("Chef",), case_sensitive=True),
    P("Puppet", "Infrastructure", "iac", ("Puppet",), case_sensitive=True),
    P("SaltStack", "Infrastructure", "iac", ("SaltStack", "Salt Stack")),
    P("Packer", "Infrastructure", None, ("HashiCorp Packer", "Packer"), case_sensitive=True),
    P("Atlantis", "Infrastructure", None, ("Atlantis",), case_sensitive=True),
    P("Spacelift", "Infrastructure", None, ("Spacelift",)),
    # --- Containers and Kubernetes add-ons (not hubs)
    P("Podman", "Compute", None, ("Podman",)),
    P("containerd", "Compute", None, ("containerd",)),
    P("Karpenter", "Compute", None, ("Karpenter",)),
    P("KEDA", "Compute", None, ("KEDA",)),
    P("Dapr", "Backend", None, ("Dapr",)),
    # --- Networking / ingress / mesh
    P("Envoy", "Edge / ingress", "load_balancer", ("Envoy Proxy", "Envoy"), case_sensitive=True),
    P("Linkerd", "Edge / ingress", None, ("Linkerd",)),
    P("Cilium", "Edge / ingress", None, ("Cilium",)),
    P("Calico", "Edge / ingress", None, ("Calico",), case_sensitive=True),
    P("CoreDNS", "Edge / ingress", "dns", ("CoreDNS", "Core DNS")),
    P("Kong", "Edge / ingress", "load_balancer", ("Kong Gateway", "Kong API Gateway", "Kong Ingress")),
    P("Cloudflare", "Edge / ingress", "cdn", ("Cloudflare", "Cloud Flare")),
    P("Akamai", "Edge / ingress", "cdn", ("Akamai",)),
    P("Fastly", "Edge / ingress", "cdn", ("Fastly",)),
    # --- Data stores
    P("CockroachDB", "Database", "database", ("CockroachDB", "Cockroach DB")),
    P("MariaDB", "Database", "database", ("MariaDB", "Maria DB")),
    P("Oracle Database", "Database", "database", ("Oracle Database", "Oracle DB")),
    P("Neo4j", "Database", "database", ("Neo4j",)),
    P("InfluxDB", "Database", "database", ("InfluxDB", "Influx DB")),
    P("TimescaleDB", "Database", "database", ("TimescaleDB",)),
    P("ClickHouse", "Database", "database", ("ClickHouse", "Click House")),
    P("Snowflake", "Database", "database", ("Snowflake",), case_sensitive=True),
    P("Databricks", "Data processing", "data_pipeline", ("Databricks",)),
    # --- Streaming and data movement
    P("Apache Pulsar", "Messaging", "queue", ("Apache Pulsar", "Pulsar"), case_sensitive=True),
    P("ZeroMQ", "Messaging", "queue", ("ZeroMQ",)),
    P("Kafka Connect", "Messaging", None, ("Kafka Connect",)),
    P("Debezium", "Data processing", "data_pipeline", ("Debezium",)),
    P("dbt", "Data processing", "data_pipeline", ("dbt",)),
    P("Airbyte", "Data processing", "data_pipeline", ("Airbyte",)),
    P("Fivetran", "Data processing", "data_pipeline", ("Fivetran",)),
    P("Temporal", "Workflow orchestration", "orchestration", ("Temporal",), case_sensitive=True),
    # --- Observability
    P("OpenTelemetry", "Observability", "monitoring", ("OpenTelemetry", "Open Telemetry", "OTel")),
    P("Zipkin", "Observability", "monitoring", ("Zipkin",)),
    P("Thanos", "Observability", "monitoring", ("Thanos",), case_sensitive=True),
    P("Grafana Mimir", "Observability", "monitoring", ("Grafana Mimir", "Mimir")),
    P("Grafana Tempo", "Observability", "monitoring", ("Grafana Tempo",)),
    P("VictoriaMetrics", "Observability", "monitoring", ("VictoriaMetrics", "Victoria Metrics")),
    P("Honeycomb", "Observability", "monitoring", ("Honeycomb",), case_sensitive=True),
    # --- Incident management
    P("incident.io", "Alerting", "alerting", ("incident.io", "incident io")),
    P("Rootly", "Alerting", "alerting", ("Rootly",)),
    P("FireHydrant", "Alerting", "alerting", ("FireHydrant", "Fire Hydrant")),
    P("Grafana OnCall", "Alerting", "alerting", ("Grafana OnCall", "Grafana On-Call")),
    P("Statuspage", "Alerting", None, ("Atlassian Statuspage", "Statuspage")),
    # --- Security and policy
    P("Snyk", "Security", None, ("Snyk",)),
    P("Aqua Security", "Security", None, ("Aqua Security",)),
    P("Falco", "Security", None, ("Falco",)),
    P("cert-manager", "Security", None, ("cert-manager",)),
    P("Grype", "Security", None, ("Grype",)),
    P("Syft", "Security", None, ("Syft",)),
    P("Cosign", "Security", None, ("Sigstore Cosign", "Cosign"), case_sensitive=True),
    P("Sigstore", "Security", None, ("Sigstore",)),
    P("Checkov", "Security", None, ("Checkov",)),
    P("tfsec", "Security", None, ("tfsec",)),
    P("Kyverno", "Security", None, ("Kyverno",)),
    P("Open Policy Agent", "Security", None, ("Open Policy Agent", "OPA"), case_sensitive=True),
    P("Dependabot", "Security", None, ("Dependabot",)),
    P("Renovate", "Security", None, ("Renovate",), case_sensitive=True),
    P("Keycloak", "Security", None, ("Keycloak",)),
    P("External Secrets Operator", "Secrets", "secrets", ("External Secrets Operator",)),
    P("Sealed Secrets", "Secrets", "secrets", ("Sealed Secrets",), case_sensitive=True),
    # --- Registries, cost, developer portal
    P("Harbor", "Container registry", "registry", ("Harbor",), case_sensitive=True),
    P("Kubecost", "Cost management", None, ("Kubecost",)),
    P("OpenCost", "Cost management", None, ("OpenCost",)),
    P("Backstage", "Developer portal", None, ("Backstage",), case_sensitive=True),
]


# Single-word names that are also ordinary words or names count only when a
# related word appears nearby (within CONTEXT_WINDOW characters). Measured on
# 1.16M words of ordinary English (NLTK Brown corpus): without this, "Harbor"
# alone matched 20 times (Pearl Harbor, Bar Harbor).
CONTEXT_WINDOW = 150
_CONTEXT_WORDS: Dict[str, str] = {
    "AWS Lambda": r"function|serverless|aws|invok|trigger|handler|api gateway|cold start|deploy",
    "Amazon Athena": r"quer|s3|sql|table|aws|data|glue",
    "Amazon S3 Glacier": r"s3|archiv|storage|backup|retriev|aws|tier",
    "Amazon Bedrock": r"model|llm|aws|claude|titan|inference|prompt|genai|embedding",
    "Bamboo": r"build|plan|pipeline|ci\b|deploy|atlassian|job",
    "Chef": r"cookbook|recipe|infra|config|server|node|provision|automat|ansible|puppet",
    "Puppet": r"manifest|module|agent|config|master|provision|automat|ansible|chef",
    "Packer": r"image|ami|build|template|golden|hashicorp|vm\b",
    "Atlantis": r"terraform|plan|pull request|\bpr\b|apply|opentofu|infra",
    "Envoy": r"proxy|sidecar|mesh|ingress|gateway|istio|traffic|load balanc",
    "Calico": r"network|polic|cni|kubernetes|cluster|pod",
    "Snowflake": r"warehouse|quer|data|table|sql|etl|dbt|analytics",
    "Apache Pulsar": r"topic|messag|stream|queue|broker|consumer|producer",
    "Temporal": r"workflow|worker|activit|orchestrat",
    "Thanos": r"prometheus|metric|retention|monitor|grafana|query",
    "Honeycomb": r"trac|observab|quer|telemetry|event|span",
    "Cosign": r"sign|image|sigstore|verif|container|supply chain",
    "Open Policy Agent": r"polic|rego|gatekeeper|admission|kubernetes",
    "Renovate": r"dependenc|pull request|\bpr\b|bot|upgrade|version|package",
    "Harbor": r"image|registry|container|chart|artifact|helm|scan|push|pull|docker",
    "Backstage": r"catalog|portal|plugin|template|developer|scaffold|docs|service",
}
_CONTEXT_RE = {name: re.compile(words, re.IGNORECASE) for name, words in _CONTEXT_WORDS.items()}


def _context_ok(product: "Product", spelling_matched: str, text: str, start: int, end: int) -> bool:
    pattern = _CONTEXT_RE.get(product.name)
    if pattern is None or len(re.split(r"[\s\-]+", spelling_matched.strip())) > 1:
        return True
    window = text[max(0, start - CONTEXT_WINDOW):start] + " " + text[end:end + CONTEXT_WINDOW]
    return bool(pattern.search(window))


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-]+", " ", (text or "").strip().lower())


def _spelling_regex(spelling: str) -> str:
    # Spaces and hyphens are interchangeable; dots are literal.
    parts = re.split(r"[\s\-]+", spelling.strip())
    return r"[\s\-]*".join(re.escape(p) for p in parts) if len(parts) > 1 else re.escape(spelling.strip())


def _build_regex() -> "re.Pattern":
    alternatives: List[Tuple[int, str]] = []
    for product in CATALOG:
        for spelling in (product.spellings or (product.name,)):
            alt = _spelling_regex(spelling)
            # Only a single word can be an ordinary English word ("Harbor",
            # "Bedrock"); "Amazon Bedrock" is unambiguous in any case, and the
            # transcript corrector writes such names in lowercase.
            if product.case_sensitive and len(re.split(r"[\s\-]+", spelling.strip())) == 1:
                alt = f"(?-i:{alt})"
            alternatives.append((len(spelling), alt))
    # Longest first so "AWS Secrets Manager" wins over "Secrets Manager".
    alternatives.sort(key=lambda a: -a[0])
    body = "|".join(a for _, a in alternatives)
    return re.compile(r"(?<![\w.\-])(" + body + r")(?![\w\-])", re.IGNORECASE)


CATALOG_REGEX = _build_regex()

# normalized spelling or name -> Product
_BY_KEY: Dict[str, Product] = {}
for _p in CATALOG:
    for _s in (_p.name,) + tuple(_p.spellings):
        _BY_KEY.setdefault(_normalize(_s), _p)


def lookup(term: str) -> Optional[Product]:
    return _BY_KEY.get(_normalize(term))


def display_name(term: str) -> Optional[str]:
    product = lookup(term)
    return product.name if product else None


def layer_terms() -> Dict[str, List[str]]:
    """layer -> lowercased names and spellings, for architecture_diagram."""
    out: Dict[str, List[str]] = {}
    for p in CATALOG:
        if p.layer:
            out.setdefault(p.layer, [])
            for s in (p.name,) + tuple(p.spellings):
                key = _normalize(s)
                if key not in out[p.layer]:
                    out[p.layer].append(key)
    return out


def category_map() -> Dict[str, str]:
    """lowercased name/spelling -> Technology Summary category."""
    out: Dict[str, str] = {}
    for p in CATALOG:
        for s in (p.name,) + tuple(p.spellings):
            out.setdefault(_normalize(s), p.category)
    return out


class ToolMatcher:
    """The original hand-written tools regex plus the catalog, behind the same
    findall()/search() interface. Every match the original regex makes is
    kept exactly; a catalog match is added where the original found nothing,
    or replaces an original match only when it is a strictly longer name
    containing it ("Kafka Connect" over "Kafka")."""

    def __init__(self, base: "re.Pattern", extra: "re.Pattern" = CATALOG_REGEX):
        self.base = base
        self.extra = extra
        self.pattern = base.pattern
        self.flags = base.flags

    def _matches(self, text: str):
        base = [(m.start(1), m.end(1), m.group(1)) for m in self.base.finditer(text or "")]
        kept = list(base)
        for m in self.extra.finditer(text or ""):
            s, e, g = m.start(1), m.end(1), m.group(1)
            product = lookup(g)
            if product is not None and not _context_ok(product, g, text, s, e):
                continue
            overlapping = [b for b in kept if b[0] < e and s < b[1]]
            if not overlapping:
                kept.append((s, e, g))
            elif all(s <= b[0] and b[1] <= e and (e - s) > (b[1] - b[0]) for b in overlapping):
                kept = [b for b in kept if b not in overlapping] + [(s, e, g)]
        return sorted(kept)

    def findall(self, text: str) -> List[str]:
        return [g for _, _, g in self._matches(text)]

    def search(self, text: str):
        found = self.base.search(text or "")
        if found:
            return found
        for m in self.extra.finditer(text or ""):
            product = lookup(m.group(1))
            if product is None or _context_ok(product, m.group(1), text, m.start(1), m.end(1)):
                return m
        return None
