"""Regression test for AWS e-commerce KT transcript classification."""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_mapper import ContextMappingPipeline
from devops_transcription import clean_transcript

TRANSCRIPT = """Hi everyone, today I will be handing over the aws e-commerce platform.

Before talking about the architecture, let me mention that this platform processes around 50,000 orders per day and is one of the company's most business critical system.

The platform consists of React front-end applications, Python fast API services, PostgreSQL, databases and Redis Cache Clear.

By the way, if you receive a PagerDuty alert at night, the first thing you should check is Grafana because most incidents are visible there immediately.

Coming back to architecture, all services run on Amazon EKS. Traffic enters through cloud front and application load balancer before reaching Kubernetes workload.

The most common production issue is database connection pool exhaustion during flash sales.

Developers own application code while the platform engineers team own Kubernetes infrastructure.

For development, code is merged into the main branch and GitHub Action Triggers, CSED pipelines automatically.

There is one important thing to remember, never modify terraform state files manually because recovering from state corruption is extremely difficult.

The disaster recovery processrequiresrestoring RDS snapshots and recreating Kubernetes workload from infrastructure as cold.

We use Prometheus, Grafana and CloudWatch for monitoring.

Another common issue is missing Kubernetes secrets. If pod fails immediately after deployment, verify secret references before investigating application code.

The staging environment variables mirrors production. However, many payments integrations are mocked in staging.

One tribal knowledge item is that cloud front cache invalidation can take longer than expected during large releases.

roll back normally performed using Helm roll back should be completed within 15 minutes.

The platform becomes extremely busy during Black Friday, month-end sales and major promotional campaigns. Deployments must be avoided during those periods.

The primary database runs on Amazon RDS and Postgres SQL with multiple az enabled.

If the system is unavailable, customers cannot place orders and warehouse fulfillment stops almost immediately.

Desklation path starts with an on-call engineer followed by the platform engineering manager and then the head of engineering.

For new team members, start by reviewing Grafana dashboards, GitHub repositories, Kubernetes namespace and deployment pipelines.

We use terraform state infrastructure provisioning, Argos CD for GitOps, deployments and Vault for secret management.

One dangerous area is that production Kubernetes cluster autoscaler configuration because one dangerous area is the production Kubernetes cluster autoscaler configuration because incorrect changes can impact the entire platform.

The architecture diagram is maintained in Confluence and should be reviewed before making intra changes.

There were a major outage last year caused by Redis memory saturation issue.

Security scanning is performed using Trevi and container images are stored in Amazon ECR.

Daily backups are retained for 30 days and disaster recovery testing is performed quarterly.

Cost optimization is important. We use spot instance in non-production environment variables schedule scaling during load traffic periods.

The platform serves customer globally and handles payments, inventory updates and shipment orchestration.

If you are unaware about my production activity, contact platform engineering before proceeding. Thank you."""

# Expected section assignments (substring match in sentence)
EXPECTED = {
    "system_overview": [
        "50,000 orders",
        "business critical",
        "react frontend",
        "customers cannot place orders",
        "serves customer globally",
        "terraform",
        "argocd",
    ],
    "architecture_reference": [
        "Amazon EKS",
        "CloudFront",
        "Confluence",
        "RDS",
    ],
    "day1_survival_checklist": [
        "new team members",
        "grafana dashboards",
    ],
    "monitoring_observability": [
        "PagerDuty alert",
        "Prometheus, Grafana and CloudWatch",
    ],
    "deployment_and_rollback": [
        "merged into the main branch",
        "Helm rollback",
    ],
    "disaster_recovery": [
        "disaster recovery",
        "RDS snapshots",
        "Daily backups",
    ],
    "common_failures": [
        "connection pool exhaustion",
        "missing Kubernetes secrets",
        "Redis memory saturation",
    ],
    "known_bad_days": [
        "Black Friday",
        "Deployments must be avoided",
    ],
    "danger_zones": [
        "never modify terraform",
        "dangerous area",
        "autoscaler",
    ],
    "ownership_escalation": [
        "Developers own",
        "escalation",
        "on-call engineer",
    ],
    "monitoring_observability": [
        "Prometheus, Grafana and CloudWatch",
    ],
    "security_controls": [
        "Security scanning",
        "Trivy",
        "ECR",
    ],
    "cost_optimization": [
        "Cost optimization",
        "spot instance",
    ],
    "open_responsibilities": [
        "contact platform engineering",
    ],
}


def build_segments(transcript: str):
    cleaned = clean_transcript(transcript)
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    segments = []
    t = 0.0
    for part in parts:
        if part.strip():
            segments.append({"text": part.strip(), "start": t, "end": t + 3, "avg_logprob": -0.3})
            t += 3
    return cleaned, segments


def test_clean_transcript_regression_fixes():
    transcript = (
        "For development, code is merged into the main branch and GitHub Action Triggers, CSED pipelines automatically. "
        "One dangerous area is that production Kubernetes cluster autoscaler configuration because one dangerous area is that production Kubernetes cluster autoscaler configuration because incorrect changes. "
        "infrastructure as cold should be reviewed."
    )

    cleaned = clean_transcript(transcript)
    normalized = cleaned.lower()

    assert "ci/cd pipelines" in normalized or "ci/cd pipeline" in normalized or "ci/cd" in normalized
    assert "infrastructure as code" in normalized or "infrastructure as code" in cleaned.lower()
    assert "Infrastructure as Code" in cleaned
    assert "because one dangerous area is that production kubernetes cluster autoscaler configuration because" not in normalized
    assert "because incorrect changes" in normalized
    assert cleaned.startswith("For") or cleaned.startswith("For development")


def run_classification():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "kt_schema_new.json")) as f:
        schema = json.load(f)["sections"]
    cleaned, segments = build_segments(TRANSCRIPT)
    pipeline = ContextMappingPipeline(schema)
    return pipeline.process("ecommerce-test", cleaned, segments)


def test_ecommerce_kt_mapping():
    kt = run_classification()
    section_texts = {}
    for sec_id, cov in kt.coverage.items():
        texts = []
        for block in (cov.blocks or []):
            for s in block.sentences:
                texts.append(s.text)
        section_texts[sec_id] = " ".join(texts).lower()

    failures = []
    for sec_id, phrases in EXPECTED.items():
        blob = section_texts.get(sec_id, "")
        for phrase in phrases:
            if phrase.lower() not in blob:
                failures.append(f"{sec_id}: missing '{phrase}'")

    # system_overview should not be a dumping ground
    overview_blob = section_texts.get("system_overview", "")
    misplaced_in_overview = [
        "escalation", "dangerous area", "helm roll back", "black friday",
        "connection pool exhaustion", "pagerduty", "security scanning",
        "disaster recovery testing", "never modify terraform",
    ]
    for phrase in misplaced_in_overview:
        if phrase.lower() in overview_blob:
            failures.append(f"system_overview: should NOT contain '{phrase}'")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("\nACTUAL DISTRIBUTION:")
        for sec_id, cov in kt.coverage.items():
            if cov.blocks:
                print(f"\n=== {cov.section_title} ({cov.status}) ===")
                for block in cov.blocks:
                    for s in block.sentences:
                        print(f"  - {s.text[:100]}")
        raise AssertionError(f"{len(failures)} classification failures")

    print("All e-commerce KT mapping checks passed.")


if __name__ == "__main__":
    test_ecommerce_kt_mapping()
