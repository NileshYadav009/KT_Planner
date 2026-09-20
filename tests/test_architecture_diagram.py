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
