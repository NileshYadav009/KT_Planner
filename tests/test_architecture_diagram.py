"""P1-9: the architecture diagram is an SVG with typed connections.

Before: an ASCII tree hung every database, queue and cache off the compute
node ("AKS ├── Azure SQL, Redis, Service Bus"), which reads as "runs inside
the cluster", and "Terraform ──► Infrastructure" pointed at a category
label. Now components are placed by what they are (managed services beside
the compute platform, third-party services outside the cloud account) and a
line is drawn only where a sentence states the connection."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from architecture_diagram import build_architecture_graph, describe_connections, render_architecture_svg

SNAPSHOTS = os.path.join(os.path.dirname(__file__), "snapshots")

TRIPWISE_COMPONENTS = ["Amazon ECS", "Fargate", "DynamoDB", "Stripe", "SendGrid", "CloudFormation", "GitHub Actions",
                       "Datadog", "PagerDuty", "AWS Secrets Manager"]
TRIPWISE_SENTENCES = [
    "The app talks to an API on Amazon ECS with Fargate.",
    "Bookings are written to DynamoDB, payments go through Stripe, and confirmation emails are sent with SendGrid.",
    "Infrastructure is defined in CloudFormation.",
    "GitHub Actions builds the container and deploys to staging automatically on merge.",
    "Datadog is our monitoring tool.",
    "Another issue is expired Stripe webhooks secrets, which silently stop payment confirmations; "
    "rotate the secret in Secrets Manager.",
]
TRIPWISE_ALIASES = {"aws secrets manager": ["secrets manager"]}

GCP_COMPONENTS = ["Pub/Sub", "Dataflow", "BigQuery", "Cloud Storage", "Cloud Run", "Memorystore", "Cloud SQL",
                  "Terraform", "Cloud Build", "Cloud Monitoring", "Opsgenie"]
GCP_SENTENCES = [
    "Devices publish to Pub/Sub.",
    "A Dataflow job enriches the events and writes them to BigQuery, and raw payloads are archived in Cloud Storage.",
    "The query API runs on Cloud Run and reads from BigQuery and a Memorystore cache.",
    "Device metadata is in Cloud SQL for Postgres.",
    "Everything is in Terraform, and Cloud Build deploys to Cloud Run on every merge to main.",
    "We watch Cloud Monitoring dashboards; the key alert is Pub/Sub oldest unacked message age over ten minutes, "
    "which pages through Opsgenie.",
    "Do not delete the BigQuery raw dataset; it is the only copy for the compliance team.",
]


def _tripwise():
    return build_architecture_graph(TRIPWISE_COMPONENTS, TRIPWISE_SENTENCES, TRIPWISE_ALIASES, "TripWise")


def _edges(graph):
    return {(e["from"], e["label"], e["to"]) for e in graph["edges"]}


def _ids(graph, key):
    return [n["id"] for n in graph[key]]


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------

def test_managed_services_sit_beside_the_platform_and_third_parties_outside_the_cloud():
    g = _tripwise()
    assert g["central"]["header"] == "Amazon ECS · Fargate" and g["central"]["label"] == "TripWise"
    assert _ids(g, "data") == ["dynamodb"]
    assert _ids(g, "external") == ["stripe", "sendgrid"]
    assert _ids(g, "delivery") == ["github-actions"]
    assert _ids(g, "operations") == ["datadog", "pagerduty", "aws-secrets-manager"]
    assert g["boundary"]["label"] == "AWS"


def _boxes(svg):
    rects = [tuple(float(v) for v in m.groups()[:4]) + (m.group(5), "stroke-dasharray" in m.group(0))
             for m in re.finditer(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" rx="\d+" '
                                  r'fill="#[0-9a-f]+" stroke="(#[0-9a-f]+)"[^>]*>', svg)]
    texts = {m.group(3): (float(m.group(1)), float(m.group(2)))
             for m in re.finditer(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*>([^<]*)</text>', svg)}
    return rects, texts


def _inside(point, rect):
    x, y = point
    rx, ry, rw, rh = rect[:4]
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def test_the_drawing_keeps_managed_services_outside_the_compute_box():
    svg = render_architecture_svg(_tripwise())
    rects, texts = _boxes(svg)
    compute = next(r for r in rects if r[4] == "#4f46e5")
    boundary = next(r for r in rects if r[5])                      # the dashed cloud account
    assert _inside(texts["TripWise"], compute)
    assert not _inside(texts["DynamoDB"], compute) and _inside(texts["DynamoDB"], boundary)
    for third_party in ("Stripe", "SendGrid"):
        assert not _inside(texts[third_party], boundary)
    assert "Provisioned with CloudFormation" in texts            # IaC names the account, not a category label


# --------------------------------------------------------------------------
# Connections: only what was said, from whoever said to do it
# --------------------------------------------------------------------------

def test_tripwise_connections_are_the_stated_ones_with_their_sentences():
    g = _tripwise()
    assert _edges(g) == {
        ("central", "writes to", "dynamodb"), ("central", "calls", "stripe"), ("central", "calls", "sendgrid"),
        ("github-actions", "deploys to", "central"), ("datadog", "monitors", "central"),
        ("aws-secrets-manager", "supplies secrets to", "central"),
    }
    rows = {(r["From"], r["Connection"], r["To"]): r["Said in the KT"] for r in describe_connections(g)}
    assert rows[("TripWise (Amazon ECS · Fargate)", "writes to", "DynamoDB")].startswith("Bookings are written to DynamoDB")
    assert rows[("CloudFormation", "provisions", "AWS")] == "Infrastructure is defined in CloudFormation."


def test_the_source_of_a_flow_is_whoever_the_sentence_says_does_it():
    g = build_architecture_graph(GCP_COMPONENTS, GCP_SENTENCES, {}, "Fieldcast")
    edges = _edges(g)
    assert ("dataflow", "writes to", "bigquery") in edges          # the pipeline writes, not the API
    assert ("central", "reads from", "bigquery") in edges          # "The query API runs on Cloud Run and reads from"
    assert ("central", "reads from", "memorystore") in edges
    assert ("central", "stores data in", "cloud-sql") in edges
    assert ("central", "archives to", "cloud-storage") in edges
    assert ("cloud-build", "deploys to", "central") in edges
    # "Devices publish to Pub/Sub": a device, not the system, so no line.
    assert not any(e["to"] == "pub-sub" for e in g["edges"])
    # A monitoring tool and a paging tool named together are not joined.
    assert not any(e["to"] == "opsgenie" or e["label"] == "alerts via" for e in g["edges"])
    assert g["boundary"] == {"label": "Google Cloud", "subtitle": "Provisioned with Terraform",
                             "quote": GCP_SENTENCES[4]}


def test_purpose_phrases_and_unstated_components():
    g = build_architecture_graph(
        ["Azure Kubernetes Service", "Azure SQL", "Azure Service Bus", "Redis", "Azure Blob Storage"],
        ["Invoices are stored in Azure SQL Database, files go to Blob Storage, and we use Azure Service Bus to queue "
         "invoice generation jobs.", "Redis is Azure Cache for Redis.", "Redis is used for caching session data."],
        {"azure blob storage": ["blob storage"]}, "Ledgerline")
    edges = _edges(g)
    assert ("central", "writes to", "azure-sql") in edges
    assert ("central", "writes to", "azure-blob-storage") in edges
    assert ("central", "publishes to", "azure-service-bus") in edges   # "Service Bus to queue invoice jobs"
    assert ("central", "caches in", "redis") in edges                 # "Redis is used for caching"


def test_no_line_is_drawn_without_a_sentence_that_states_it():
    g = build_architecture_graph(["Amazon EKS", "Amazon RDS", "Redis", "Kafka", "Stripe"], [], {}, None)
    assert g["edges"] == []
    assert _ids(g, "data") == ["amazon-rds", "redis", "kafka"] and _ids(g, "external") == ["stripe"]
    assert g["central"]["label"] == "Application"


def test_users_and_the_request_path_only_with_a_customer_facing_entry_point():
    g = build_architecture_graph(["Route 53", "CloudFront", "Application Load Balancer", "Amazon EKS", "Spring Boot",
                                  "Aurora PostgreSQL"], [], {}, "Shop")
    assert [n["label"] for n in g["entry"]] == ["Users", "Route 53", "CloudFront", "Application Load Balancer"]
    assert g["central"]["caption"] == "Spring Boot"
    backend = build_architecture_graph(["Google Kubernetes Engine", "BigQuery", "Airflow"], [], {}, None)
    assert backend["entry"] == []


def test_the_platform_is_named_by_its_most_specific_product():
    g = build_architecture_graph(["Kubernetes", "Amazon EKS", "Amazon RDS"], [], {}, None)
    assert g["central"]["header"] == "Amazon EKS"
    plain = build_architecture_graph(["Kubernetes", "PostgreSQL"], [], {}, None)
    assert plain["central"]["header"] == "Kubernetes" and plain["boundary"]["label"] == "Platform"


def test_a_managed_offering_and_its_engine_are_one_component():
    g = build_architecture_graph(["Amazon EKS", "Kafka", "Amazon MSK", "Amazon RDS", "PostgreSQL"], [], {}, None)
    assert [n["label"] for n in g["data"]] == ["Amazon RDS", "Kafka (Amazon MSK)"]


def test_nothing_to_draw_returns_none():
    assert build_architecture_graph([], [], {}, None) is None
    assert build_architecture_graph(["Kubernetes"], [], {}, None) is None
    assert build_architecture_graph(["We", "Reviewed"], [], {}, None) is None


# --------------------------------------------------------------------------
# The SVG itself
# --------------------------------------------------------------------------

def test_labels_are_escaped_and_the_svg_is_self_contained():
    g = build_architecture_graph(["Amazon EKS", "DynamoDB"], [], {}, '<script>alert(1)</script>"&')
    svg = render_architecture_svg(g)
    assert "<script>" not in svg and "&lt;script&gt;" in svg
    assert "href" not in svg and "http://www.w3.org/2000/svg" in svg and svg.count("http") == 1


def test_the_tripwise_drawing_matches_its_snapshot():
    """The layout is deterministic. If a deliberate layout change breaks this,
    look at the new drawing, then rerun with UPDATE_SNAPSHOTS=1."""
    svg = render_architecture_svg(_tripwise())
    assert svg == render_architecture_svg(_tripwise())
    path = os.path.join(SNAPSHOTS, "architecture_tripwise.svg")
    if os.getenv("UPDATE_SNAPSHOTS") == "1" or not os.path.exists(path):
        os.makedirs(SNAPSHOTS, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg)
    with open(path, encoding="utf-8") as fh:
        assert svg == fh.read()
