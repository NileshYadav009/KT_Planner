"""Tests for architecture_diagram.py's build_architecture_flow_diagram() —
turns a flat list of detected component names into a top-down "mental
model" tree diagram, using a coarse layer classification rather than
parsing arbitrary transcript sentences for relationship language.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from architecture_diagram import build_architecture_flow_diagram


def test_full_chain_with_hub_fanout_and_registry():
    components = [
        "FastAPI", "Terraform", "GitHub Actions", "Amazon EKS", "React",
        "CloudFront", "Application Load Balancer", "Kubernetes", "Amazon RDS",
        "PostgreSQL", "Redis", "Amazon SQS", "Amazon ECR", "Prometheus",
        "Grafana", "CloudWatch", "PagerDuty", "Vault", "Helm",
    ]
    diagram = build_architecture_flow_diagram(components)
    assert diagram is not None

    lines = diagram.splitlines()
    # Main chain, in order: Customer -> React -> CloudFront -> ALB -> EKS
    assert lines[0] == "Customer"
    assert "React" in diagram
    chain_order = [l for l in lines if l in ("Customer", "React", "CloudFront", "Application Load Balancer", "Amazon EKS")]
    assert chain_order == ["Customer", "React", "CloudFront", "Application Load Balancer", "Amazon EKS"]

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
    assert diagram.strip().endswith("Amazon ECR\n └── Container images")

    # Supporting infrastructure (monitoring/alerting/IaC/CI-CD/secrets) has
    # no place in a request-flow diagram — must not appear at all.
    for noise in ("Terraform", "GitHub Actions", "Prometheus", "Grafana", "CloudWatch", "PagerDuty", "Vault", "Helm"):
        assert noise not in diagram


def test_no_compute_hub_falls_back_to_flat_chain_instead_of_inventing_a_branch_point():
    # FastAPI + Redis are named but no orchestration/compute term (Kubernetes,
    # EKS, Docker, Rancher) is — there's nothing to legitimately fan out
    # FROM, so this must not fabricate a hub that was never stated.
    components = ["React", "FastAPI", "Redis"]
    diagram = build_architecture_flow_diagram(components)
    assert diagram is not None
    assert "├──" not in diagram
    assert "└──" not in diagram or "Container images" not in diagram
    lines = [l.strip() for l in diagram.splitlines() if l.strip() and l.strip() not in ("│", "▼")]
    assert lines == ["Customer", "React", "FastAPI", "Redis"]


def test_registry_only_renders_without_a_request_flow_chain():
    diagram = build_architecture_flow_diagram(["Amazon ECR"])
    assert diagram == "Amazon ECR\n └── Container images"


def test_no_recognizable_layer_terms_returns_none():
    # Only supporting-infrastructure terms — nothing resembling a request
    # flow to draw. Must not fabricate an empty or misleading diagram.
    assert build_architecture_flow_diagram(["Terraform", "Jenkins", "Prometheus"]) is None


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
