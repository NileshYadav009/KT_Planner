"""P0-10: the tables an incident responder reads first must not print wrong
rows. Each case below is a row the audit found in a real PDF."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from renderers.sections.ownership_escalation import _explicit_ownership_statements


@pytest.mark.parametrize("text,expected", [
    ("The mobile team owns the app, and our squad owns the backend.",
     [("Mobile team", "The app"), ("Our squad", "The backend")]),
    ("The platform team owns the cluster; we own the application.",
     [("Platform team", "The cluster"), ("Our team (outgoing owner)", "The application")]),
    ("The payments team owns the service. Actually the platform team owns it since the reorg.",
     [("Payments team", "The service"), ("Platform team", "The service (since the reorg)")]),
    ("Developers own the application code while the platform engineering team owns the Kubernetes infrastructure.",
     [("Developers", "The application code"), ("Platform engineering team", "The Kubernetes infrastructure")]),
    ("Who owns the database?", []),
])
def test_ownership_rows_are_one_owner_per_clause(text, expected):
    assert _explicit_ownership_statements({"coverage_content": [text]}) == expected


def test_a_sentence_listing_environments_is_not_one_environments_characteristic():
    from field_populator import _extract_by_semantic_scored

    field = {"id": "staging_notes", "label": "Staging characteristics", "type": "text"}
    sentences = ["We have three environments: dev, staging, and production.",
                 "Staging uses Stripe test keys, so you can book freely there."]
    value, _ = _extract_by_semantic_scored(field, sentences, model=None, own_identity_word="staging",
                                           other_identity_words=["production", "non-production"])
    assert value == "Staging uses Stripe test keys, so you can book freely there."


def test_week_plan_sentences_go_under_the_week_they_name():
    from field_populator import _redistribute_weeks

    fields = {
        "week1": {"value": "In your first week, shadow the on-call and read the runbooks. By week three you should own releases.",
                  "confidence": 0.65, "source": "semantic"},
        "week3": {"value": "", "confidence": 0.0, "source": "unfilled"},
    }
    _redistribute_weeks(fields)
    assert fields["week1"]["value"] == "In your first week, shadow the on-call and read the runbooks."
    assert fields["week3"]["value"] == "By week three you should own releases."


@pytest.mark.parametrize("sentence,expected", [
    ("Service accounts use least privilege and VPC Service Controls protect BigQuery.", "VPC Service Controls"),
    ("The cluster runs in an Amazon VPC with private subnets.", "Amazon VPC"),
    ("Each environment has its own VPC.", "VPC"),
])
def test_vpc_is_not_labelled_amazon_on_other_clouds(sentence, expected):
    import component_catalog as cc
    from field_populator import PATTERN_EXTRACTORS

    names = [cc.display_name(t) or t for t in PATTERN_EXTRACTORS["tools"].findall(sentence)]
    assert expected in names
    if expected != "Amazon VPC":
        assert "Amazon VPC" not in names


def test_cache_layer_row_lists_only_caches():
    from renderers.sections.system_overview import render

    section = {"id": "system_overview", "title": "System Overview", "coverage_content": [], "fields": {
        "cache_layer": {"value": "Invoices are stored in Azure SQL Database, files go to Blob Storage, "
                                 "and we use Azure Service Bus to queue jobs. Redis is Azure Cache for Redis.",
                        "source": "semantic", "confidence": 0.65}}}
    rows = [r for b in render(section)["blocks"] if b["type"] == "TechnologyGrid" for r in b["rows"]]
    cache_rows = [r for r in rows if r["label"] == "Cache Layer"]
    assert cache_rows and all("Blob" not in r["value"] and "SQL" not in r["value"] for r in cache_rows)
