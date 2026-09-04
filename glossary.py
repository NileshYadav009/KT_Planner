"""
Glossary and conservative ASR repair utilities.

Provides phrase- and token-level corrections for common technical terms
and acronyms. Designed to run only when sentence-level audio confidence is
low to avoid hallucination.
"""
from typing import Tuple
import json
import os
import re
import logging
from difflib import SequenceMatcher, get_close_matches

logger = logging.getLogger(__name__)

# Default technical glossary and phrase corrections (conservative).
#
# This is the bootstrap content written to glossary.json the first time it's
# created on a fresh machine (see load_glossary/save_glossary below). It grows
# from there via the self-learning loop: vocabulary_learning.py detects candidate
# terms from real transcripts, and scripts/review_vocabulary.py lets a human
# approve them into this same file — see REPOSITORY_AUDIT.md's Phase 4 notes.
DEFAULT_GLOSSARY = {
    "terms": [
        # Core web/infra
        "REST", "HTTP", "HTTPS", "API", "SQL", "gRPC", "GraphQL", "WebSocket",
        "TCP", "UDP", "DNS", "CDN", "VPN", "TLS", "SSL", "async", "streaming",
        "event", "service", "services", "microservice", "microservices",
        # Containers & orchestration
        "Kubernetes", "Docker", "Podman", "containerd", "Helm", "Pod",
        "Deployment", "StatefulSet", "DaemonSet", "Ingress", "ConfigMap",
        "Namespace", "etcd", "kubectl", "Kustomize", "Rancher", "OpenShift",
        "Minikube", "Cilium", "Calico",
        # AWS
        "AWS", "EC2", "S3", "RDS", "Lambda", "ECS", "EKS", "Fargate",
        "CloudFront", "Route53", "IAM", "VPC", "CloudWatch", "SQS", "SNS",
        "DynamoDB", "ElastiCache", "ECR", "CloudFormation", "Aurora",
        "CloudTrail", "KMS", "WAF", "STS", "CodePipeline", "CodeBuild",
        "CodeDeploy", "Secrets Manager", "Systems Manager", "Step Functions",
        "EventBridge", "SageMaker", "Cognito",
        # Azure / GCP
        "Azure", "AKS", "GCP", "GKE", "BigQuery", "Cosmos DB",
        "Azure DevOps", "Azure Functions", "Azure Monitor", "Cloud Run",
        "Cloud Functions", "Cloud Build", "Pub/Sub", "BigTable",
        # CI/CD & IaC
        "Jenkins", "GitLab", "GitHub", "GitHub Actions", "CircleCI",
        "Terraform", "Ansible", "Chef", "Puppet", "ArgoCD", "Helm chart",
        "CI/CD", "GitOps", "Spinnaker", "Tekton", "Flux", "Pulumi", "Packer",
        # Observability
        "Grafana", "Prometheus", "Datadog", "Splunk", "Elasticsearch",
        "Logstash", "Kibana", "OpenTelemetry", "Jaeger", "PagerDuty",
        "OpsGenie", "New Relic", "Sentry", "Fluentd", "Loki",
        # Messaging / data
        "Kafka", "RabbitMQ", "Redis", "PostgreSQL", "MySQL", "MongoDB",
        "Cassandra", "Snowflake",
        # Security
        "Vault", "Trivy", "SonarQube", "Snyk", "OWASP", "OAuth", "JWT",
        "RBAC", "MFA", "Okta", "Keycloak", "SIEM", "Falco",
        # Networking
        "NGINX", "HAProxy", "Envoy", "Istio", "Linkerd", "Consul", "BGP",
        # Concepts
        "SRE", "SLA", "SLO", "SLI", "RTO", "RPO", "rollback", "failover",
        "on-call", "runbook", "postmortem", "canary deployment",
        "blue-green deployment", "MTTR", "MTTA", "MTBF", "DORA metrics",
        "error budget", "toil", "chaos engineering", "zero trust",
        "least privilege", "blast radius", "FinOps",
        "trunk-based development", "monorepo", "technical debt",
        "incident commander", "war room", "game day",
    ],
    # Known phrase-level mis-transcriptions -> corrections
    "phrase_corrections": {
        # Example mapping covering the user's example transcription mistake
        "coffee for a sink event screaming between surfaces": "Kafka for async event streaming between services",
        # Other conservative corrections
        "coffee for a sink": "Kafka for async",
        "sink event": "sync event",
        "screaming": "streaming",
        "surfaces": "services"
    },
    # Acronym canonical forms
    "acronyms": {
        "ci": "CI",
        "cd": "CD",
        "ci/cd": "CI/CD",
        "api": "API",
        "rest": "REST",
        "s3": "S3",
        "ec2": "EC2",
        "rds": "RDS",
        "vpc": "VPC",
        "iam": "IAM",
        "ecs": "ECS",
        "eks": "EKS",
        "aks": "AKS",
        "gke": "GKE",
        "sqs": "SQS",
        "sns": "SNS",
        "kafka": "Kafka",
        "sre": "SRE",
        "sla": "SLA",
        "slo": "SLO",
        "sli": "SLI",
        "rto": "RTO",
        "rpo": "RPO",
        "rbac": "RBAC",
        "mfa": "MFA",
        "jwt": "JWT",
        "oauth": "OAuth",
        "tls": "TLS",
        "ssl": "SSL",
        "vpn": "VPN",
        "cdn": "CDN",
        "dns": "DNS",
        "kms": "KMS",
        "waf": "WAF",
        "sts": "STS",
        "acm": "ACM",
        "msk": "MSK",
        "acr": "ACR",
        "aci": "ACI",
        "idp": "IDP",
        "apm": "APM",
        "rum": "RUM",
        "opa": "OPA",
        "cap": "CAP",
        "bgp": "BGP",
        "vpa": "VPA",
        "pdb": "PDB",
        "mttr": "MTTR",
        "mtta": "MTTA",
        "mtbf": "MTBF",
        "siem": "SIEM",
        "iac": "IaC",
        "devex": "DevEx",
        "grpc": "gRPC",
        "tcp": "TCP",
        "udp": "UDP",
        "http": "HTTP",
        "https": "HTTPS",
        "sql": "SQL",
    }
}


def load_glossary(path: str = "glossary.json") -> dict:
    """Load a user-provided glossary from workspace if present, else default."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded glossary from {path}")
                return data
    except Exception as e:
        logger.warning(f"Failed to load glossary {path}: {e}")
    return DEFAULT_GLOSSARY


GLOSSARY = load_glossary()

# GLOSSARY above is only a snapshot from whenever this module was first
# imported in *this* process. save_glossary() writing glossary.json from one
# process (e.g. scripts/review_vocabulary.py) does not update another
# already-running process's copy of this module — each Python process holds
# its own private in-memory state, even for the same imported file. That
# matters here specifically because a long-lived server process (main.py) and
# a short-lived review-CLI process are expected to be different processes.
# get_glossary() below re-reads from disk when the file's mtime has moved past
# what this process last saw, so an approval made by the CLI takes effect in
# the running server without a restart. Internal glossary.py functions and
# devops_transcription.py both call get_glossary() rather than reading the
# bare GLOSSARY name for this reason.
_GLOSSARY_PATH_LOADED = "glossary.json"
_GLOSSARY_MTIME = os.path.getmtime(_GLOSSARY_PATH_LOADED) if os.path.exists(_GLOSSARY_PATH_LOADED) else None


def get_glossary(path: str = "glossary.json") -> dict:
    """Return the current glossary, transparently reloading from disk if the
    file has changed since this process last read it (see note above)."""
    global GLOSSARY, _GLOSSARY_PATH_LOADED, _GLOSSARY_MTIME

    try:
        mtime = os.path.getmtime(path) if os.path.exists(path) else None
    except Exception:
        mtime = None

    if path != _GLOSSARY_PATH_LOADED or mtime != _GLOSSARY_MTIME:
        GLOSSARY = load_glossary(path)
        _GLOSSARY_PATH_LOADED = path
        _GLOSSARY_MTIME = mtime

    return GLOSSARY


def save_glossary(data: dict, path: str = "glossary.json") -> bool:
    """Save glossary back to workspace path. Returns True on success."""
    global GLOSSARY, _GLOSSARY_PATH_LOADED, _GLOSSARY_MTIME
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # update in-memory cache + the mtime bookkeeping get_glossary() uses,
        # so a process that both saves and immediately reads sees its own write
        GLOSSARY = data
        _GLOSSARY_PATH_LOADED = path
        _GLOSSARY_MTIME = os.path.getmtime(path)
        logger.info(f"Saved glossary to {path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save glossary {path}: {e}")
        return False


AMBIGUITY_NEARBY = ["customer", "interaction", "physical", "bucket", "container", "box", "person", "human", "client"]


def detect_ambiguous_usage(text: str) -> list:
    """Detect glossary terms used near ambiguous non-technical words.

    Returns list of warning strings. Conservative: only flags when a glossary
    token appears within 3 tokens of an ambiguous word.
    """
    warnings = []
    if not text:
        return warnings
    current_glossary = get_glossary()
    toks = re.findall(r"\w+", text.lower())
    for i, tok in enumerate(toks):
        # check if token matches a glossary term or acronym
        for term in (current_glossary.get("terms", []) + list(current_glossary.get("acronyms", {}).keys())):
            if tok == term.lower():
                # window check for ambiguous neighbors
                start = max(0, i - 3)
                end = min(len(toks), i + 4)
                window = toks[start:end]
                for amb in AMBIGUITY_NEARBY:
                    if amb in window:
                        warnings.append(f"Glossary term '{term}' appears near ambiguous word '{amb}'")
                        break
    # deduplicate
    return list(dict.fromkeys(warnings))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def apply_glossary_corrections(text: str, sentence_confidence: float, min_confidence_for_correction: float = 0.6) -> Tuple[str, bool]:
    """
    Apply conservative glossary corrections.

    - Phrase-corrections are applied if the sentence confidence is below the
      `min_confidence_for_correction` threshold.
    - Token-level normalization will canonicalize known acronyms and exact
      term matches. Token fuzzy replacement is conservative and uses
      similarity thresholds.

    Returns: (new_text, changed_flag)
    """
    original = text
    text = _normalize_whitespace(text)
    current_glossary = get_glossary()

    changed = False

    # Only apply aggressive phrase-corrections when confidence is low
    if sentence_confidence < min_confidence_for_correction:
        # Try direct phrase corrections first (longest-first)
        phrase_map = current_glossary.get("phrase_corrections", {})
        # Sort keys by length to match longer phrases first
        for bad_phrase in sorted(phrase_map.keys(), key=len, reverse=True):
            pattern = re.compile(re.escape(bad_phrase), re.IGNORECASE)
            if pattern.search(text):
                replacement = phrase_map[bad_phrase]
                text = pattern.sub(replacement, text)
                changed = True

    # Token-level canonicalization for acronyms and technical terms
    tokens = text.split()
    acronyms = current_glossary.get("acronyms", {})
    terms = current_glossary.get("terms", [])

    def canonical_for_token(tok: str) -> str:
        lowered = tok.lower()
        if lowered in acronyms:
            return acronyms[lowered]
        # Exact term match (case-insensitive)
        for t in terms:
            if lowered == t.lower():
                return t
        # Do NOT perform fuzzy reinterpretation. If ambiguity exists, preserve original token.
        return tok

    new_tokens = []
    for tok in tokens:
        canon = canonical_for_token(tok)
        if canon != tok:
            changed = True
        new_tokens.append(canon)

    new_text = " ".join(new_tokens)

    # Final normalization and return
    new_text = _normalize_whitespace(new_text)
    return new_text, changed
