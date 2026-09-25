"""Tests for architecture_diagram.py's build_architecture_flow_diagram() —
turns a flat list of detected component names into a top-down "mental
model" diagram: the main request-flow tree, plus separate supporting-
infrastructure flows (CI/CD, IaC, secrets, observability, alerting) — using
a coarse layer classification rather than parsing arbitrary transcript
sentences for relationship language.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from architecture_diagram import build_architecture_flow_diagram


def test_full_diagram_covers_main_flow_and_every_supporting_section():
    components = [
        "FastAPI", "Terraform", "GitHub Actions", "Amazon EKS", "React",
        "CloudFront", "Application Load Balancer", "Amazon RDS",
        "PostgreSQL", "Redis", "Amazon SQS", "Amazon ECR", "Prometheus",
        "Grafana", "CloudWatch", "PagerDuty", "Vault", "ArgoCD",
    ]
    diagram = build_architecture_flow_diagram(components)
    assert diagram is not None

    lines = diagram.splitlines()
    # Main chain, in order: Customer -> React -> CloudFront -> ALB -> EKS.
    # "Amazon EKS" legitimately appears a second time later, as the
    # destination of the CI/CD flow section below — only the first 5
    # matches (the main chain itself) need to be in this exact order.
    assert lines[0] == "Customer"
    chain_order = [l for l in lines if l in ("Customer", "React", "CloudFront", "Application Load Balancer", "Amazon EKS")]
    assert chain_order[:5] == ["Customer", "React", "CloudFront", "Application Load Balancer", "Amazon EKS"]

    # Fan-out from the EKS hub: service + database + cache + queue
    assert "├── FastAPI" in diagram
    assert "├── Amazon RDS" in diagram
    assert "├── Redis" in diagram
    assert "└── Amazon SQS" in diagram

    # PostgreSQL is a synonym for the same "database" layer as Amazon RDS —
    # first-seen (Amazon RDS, which appears earlier in the input list) wins,
    # so PostgreSQL must not appear as a SEPARATE fan-out branch.
    assert "PostgreSQL" not in diagram

    # Registry shown separately from the request-flow chain/fan-out.
    assert "Amazon ECR\n └── Container images" in diagram

    # Supporting infrastructure now gets its OWN section — never force-fit
    # into the request-flow chain/fan-out above, but not silently dropped
    # either.
    assert "GitHub Actions" in diagram and "CI" in diagram and "ArgoCD" in diagram
    assert "Terraform ──► Infrastructure" in diagram
    assert "Vault ──► Secrets" in diagram
    assert "Prometheus, Grafana, CloudWatch ──► Observability" in diagram
    assert "PagerDuty ──► Alerting" in diagram


def test_no_compute_hub_falls_back_to_flat_chain_instead_of_inventing_a_branch_point():
    # FastAPI + Redis are named but no orchestration/compute term (Kubernetes,
    # EKS, Docker, Rancher) is — there's nothing to legitimately fan out
    # FROM, so this must not fabricate a hub that was never stated.
    components = ["React", "FastAPI", "Redis"]
    diagram = build_architecture_flow_diagram(components)
    assert diagram is not None
    assert "├──" not in diagram
    lines = [l.strip() for l in diagram.splitlines() if l.strip() and l.strip() not in ("│", "▼")]
    assert lines == ["Customer", "React", "FastAPI", "Redis"]


def test_registry_only_renders_without_a_request_flow_chain():
    diagram = build_architecture_flow_diagram(["Amazon ECR"])
    assert diagram == "Amazon ECR\n └── Container images"


def test_supporting_infrastructure_only_still_renders_its_own_sections():
    # These terms have no request-flow position, but they ARE recognized
    # supporting-infrastructure layers (iac/cicd/monitoring) — must render
    # their own flows rather than returning None just because there's no
    # frontend/compute/data-layer chain to draw.
    diagram = build_architecture_flow_diagram(["Terraform", "Jenkins", "Prometheus"])
    assert diagram is not None
    assert "Customer" not in diagram
    assert "Terraform ──► Infrastructure" in diagram
    assert "Prometheus ──► Observability" in diagram
    assert "Jenkins" in diagram and "CI" in diagram


def test_bare_compute_hub_with_no_chain_or_fanout_is_not_rendered_as_a_floating_node():
    # Regression test for a real live KT: a transcript naming only
    # Kubernetes, Terraform, and Jenkins (no frontend/CDN/LB, no fan-out
    # services/databases) produced a diagram with a standalone "Kubernetes"
    # line with zero arrows or context, immediately followed by a separate
    # "Jenkins -> CI -> Kubernetes" CI/CD flow ending at the exact same
    # node — the same fact rendered twice as if it were two disconnected
    # pieces of information. A lone node with no relationships conveys
    # nothing the flat Architecture Knowledge list doesn't already say, so
    # the main-flow section should be omitted in that case; the CI/CD flow
    # (and any other supporting section) still renders normally.
    diagram = build_architecture_flow_diagram(["Kubernetes", "Terraform", "Jenkins"])
    assert diagram is not None
    lines = [l.strip() for l in diagram.splitlines() if l.strip()]
    # "Kubernetes" only legitimately appears as the CI/CD flow's
    # destination, never as an isolated leading line with no arrow.
    assert lines[0] != "Kubernetes"
    assert "Jenkins" in diagram and "CI" in diagram and "Kubernetes" in diagram
    assert "Terraform ──► Infrastructure" in diagram


def test_dependency_layers_are_visually_distinguished_from_the_hosted_workload():
    # A compute hub with both a hosted service AND dependencies (database/
    # cache/queue) used to render all four as identical "├──" tree children
    # of the same hub — visually implying the database/cache/queue were
    # hosted INSIDE the compute node, when really only the service (the
    # application workload) is hosted there; the database/cache/queue are
    # dependencies OF that workload, not of the cluster itself. A labeled
    # sub-group must separate the two.
    components = ["FastAPI", "Amazon EKS", "Amazon RDS", "Redis", "Amazon SQS"]
    diagram = build_architecture_flow_diagram(components)
    assert diagram is not None
    assert "(workload dependencies)" in diagram

    lines = diagram.splitlines()
    hosted_idx = next(i for i, l in enumerate(lines) if "FastAPI" in l)
    label_idx = next(i for i, l in enumerate(lines) if "(workload dependencies)" in l)
    dependency_idx = next(i for i, l in enumerate(lines) if "Amazon RDS" in l)
    # The label appears after the hosted workload and before the
    # dependencies it's introducing.
    assert hosted_idx < label_idx < dependency_idx


def test_dependency_label_omitted_when_no_hosted_workload_is_named():
    # No service/app-framework term named (only a database) — there's
    # nothing to distinguish the dependency FROM, so no label should be
    # added; this must render exactly as it did before the hosted/
    # dependency split.
    components = ["Application Load Balancer", "Amazon EKS", "Amazon RDS"]
    diagram = build_architecture_flow_diagram(components)
    assert diagram is not None
    assert "(workload dependencies)" not in diagram
    assert "├── Amazon RDS" in diagram or "└── Amazon RDS" in diagram


def test_no_recognizable_layer_terms_returns_none():
    # Nothing here maps to any known layer at all. Must not fabricate an
    # empty or misleading diagram.
    assert build_architecture_flow_diagram(["Helm", "SonarQube", "Nexus"]) is None


def test_empty_components_returns_none():
    assert build_architecture_flow_diagram([]) is None
    assert build_architecture_flow_diagram(None) is None


def test_partial_chain_starts_from_whatever_layers_are_actually_present():
    # No frontend or CDN named — chain should just start from what IS there.
    components = ["Application Load Balancer", "Amazon EKS", "Amazon RDS"]
    diagram = build_architecture_flow_diagram(components)
    lines = diagram.splitlines()
    assert lines[0] == "Customer"
    assert "React" not in diagram
    assert "CloudFront" not in diagram
    assert "Application Load Balancer" in diagram
    assert "├── Amazon RDS" in diagram or "└── Amazon RDS" in diagram


def test_cicd_flow_degrades_gracefully_when_only_partially_named():
    # Only a CI/CD tool named, no GitOps tool, no registry, no compute hub
    # — must still draw the short flow it actually has, not nothing.
    diagram = build_architecture_flow_diagram(["GitHub Actions"])
    assert diagram is not None
    assert "GitHub Actions" in diagram
    assert "CI" in diagram


def test_azure_vocabulary_produces_a_full_diagram():
    components = [
        "Angular", "Azure Front Door", "Application Gateway",
        "Azure Kubernetes Service", "Azure SQL", "Azure Service Bus",
        "Azure Container Registry", "Azure DevOps", "Flux",
        "Azure Key Vault", "Bicep", "Azure Monitor", "Application Insights",
        "PagerDuty",
    ]
    diagram = build_architecture_flow_diagram(components)
    assert diagram is not None
    lines = diagram.splitlines()
    # "Azure Kubernetes Service" legitimately appears again as the CI/CD
    # flow's destination — only the first 5 matches (the main chain) need
    # to be in this exact order.
    chain_order = [l for l in lines if l in ("Customer", "Angular", "Azure Front Door", "Application Gateway", "Azure Kubernetes Service")]
    assert chain_order[:5] == ["Customer", "Angular", "Azure Front Door", "Application Gateway", "Azure Kubernetes Service"]
    assert "├── Azure SQL" in diagram or "└── Azure SQL" in diagram
    assert "├── Azure Service Bus" in diagram or "└── Azure Service Bus" in diagram
    assert "Azure Container Registry\n └── Container images" in diagram
    assert "Azure DevOps" in diagram and "Flux" in diagram
    assert "Bicep ──► Infrastructure" in diagram
    assert "Azure Key Vault ──► Secrets" in diagram
    assert "Azure Monitor, Application Insights ──► Observability" in diagram
    assert "PagerDuty ──► Alerting" in diagram


def test_gcp_vocabulary_produces_a_full_diagram_without_fabricating_a_customer_entry():
    # Regression test for a real live bug: a GCP data/ML platform transcript
    # (no frontend/CDN/load-balancer ever mentioned — it's not a
    # customer-facing web system) produced a diagram that started with
    # "Customer -> Kubernetes" even though no customer-facing entry point
    # was ever named, and the GCP-specific vocabulary (Pub/Sub, BigQuery,
    # GKE, Dataflow, Airflow, Vertex AI, Artifact Registry, Secret Manager,
    # Cloud Monitoring/Logging) was entirely unrecognized (only GitHub
    # Actions/Terraform/Kubernetes/Grafana showed up at all).
    components = [
        "GitHub Actions", "Terraform", "Google Kubernetes Engine", "Grafana",
        "ArgoCD", "Artifact Registry", "Secret Manager", "Pub/Sub",
        "Dataflow", "BigQuery", "Airflow", "Vertex AI", "Cloud Monitoring",
        "Cloud Logging", "PagerDuty",
    ]
    diagram = build_architecture_flow_diagram(components)
    assert diagram is not None

    # No customer-facing layer (frontend/CDN/load-balancer) was named, so
    # no "Customer" root should be fabricated.
    assert "Customer" not in diagram

    assert "Google Kubernetes Engine" in diagram
    assert "├── BigQuery" in diagram or "└── BigQuery" in diagram
    assert "├── Pub/Sub" in diagram or "└── Pub/Sub" in diagram
    assert "Artifact Registry\n └── Container images" in diagram
    assert "GitHub Actions" in diagram and "ArgoCD" in diagram
    assert "Terraform ──► Infrastructure" in diagram
    assert "Secret Manager ──► Secrets" in diagram
    assert "Grafana, Cloud Monitoring, Cloud Logging ──► Observability" in diagram
    assert "PagerDuty ──► Alerting" in diagram
    assert "Airflow ──► Workflow Orchestration" in diagram
    assert "Vertex AI ──► Machine Learning" in diagram
    assert "Dataflow ──► Data Processing" in diagram


def test_compute_hub_prefers_specific_branded_name_over_generic_kubernetes():
    # Regression test: a transcript that names both a specific managed-K8s
    # product ("Amazon EKS") AND generic "Kubernetes" (very common — e.g.
    # "Amazon EKS... Kubernetes workloads...") used to label the hub with
    # whichever term the scan happened to see FIRST, which is order-
    # dependent and inconsistent run to run. Must always prefer the
    # specific/branded name when both are present, regardless of order.
    components_specific_first = ["Amazon EKS", "React", "Kubernetes", "Redis"]
    components_generic_first = ["Kubernetes", "React", "Amazon EKS", "Redis"]
    for components in (components_specific_first, components_generic_first):
        diagram = build_architecture_flow_diagram(components)
        assert "Amazon EKS" in diagram
        lines = [l.strip() for l in diagram.splitlines()]
        assert "Kubernetes" not in lines  # bare "Kubernetes" must not win the hub slot


def test_customer_entry_only_shown_with_real_evidence_of_a_customer_facing_layer():
    # A compute hub with backing services but no frontend/CDN/load-balancer
    # ever named (e.g. an internal service or data platform) must not
    # imply a customer request path that was never described.
    diagram = build_architecture_flow_diagram(["Kubernetes", "PostgreSQL", "Redis"])
    assert diagram is not None
    assert "Customer" not in diagram
    assert diagram.splitlines()[0] == "Kubernetes"
