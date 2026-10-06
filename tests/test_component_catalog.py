"""component_catalog: products added in one place reach the tool recogniser,
the diagram and the Technology Summary, without changing anything the
original hand-written lists already recognised."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import component_catalog as cc
from architecture_diagram import _LAYER_TERMS, build_architecture_graph
from field_populator import PATTERN_EXTRACTORS
from knowledge.knowledge_builder import _canonicalize_component_term
from renderers.sections.system_overview import _TECH_CATEGORY_MAP

TOOLS = PATTERN_EXTRACTORS["tools"]


def test_catalog_entries_are_well_formed():
    layers = set(_LAYER_TERMS)
    names = [p.name.lower() for p in cc.CATALOG]
    assert len(names) == len(set(names)), "duplicate product names"
    for p in cc.CATALOG:
        assert p.category and p.spellings, p.name
        assert p.layer is None or p.layer in layers, (p.name, p.layer)


@pytest.mark.parametrize("sentence,expected", [
    ("Legacy batch jobs still run on EC2.", ["EC2"]),
    ("Code lives in GitLab.", ["GitLab"]),
    ("Change events flow through Kafka Connect into ClickHouse.", ["Kafka Connect", "ClickHouse"]),
    ("Secrets are kept in AWS Secrets Manager.", ["AWS Secrets Manager"]),
    ("Images are stored in Harbor and signed with Cosign.", ["Harbor", "Cosign"]),
    ("Infrastructure is written with OpenTofu and Terragrunt.", ["OpenTofu", "Terragrunt"]),
    ("Incidents are run in incident.io.", ["incident.io"]),
])
def test_new_products_are_recognised(sentence, expected):
    assert TOOLS.findall(sentence) == expected


@pytest.mark.parametrize("sentence", [
    "We had lunch by the harbor.",
    "This is only a temporary workaround.",
    "The puppet show and the chef's special.",
    "Our Hong Kong office handles it.",
    "Use a lambda expression here.",
    "Work happens backstage before the release.",
])
def test_ordinary_words_are_not_products(sentence):
    assert TOOLS.findall(sentence) == []


@pytest.mark.parametrize("sentence,expected", [
    # Longer existing names keep winning over a shorter catalog name.
    ("GitHub Actions builds the image.", ["GitHub Actions"]),
    ("GitLab CI runs the tests.", ["GitLab CI"]),
    # Existing matches are unchanged.
    ("Kafka, Redis and PostgreSQL on Amazon EKS.", ["Kafka", "Redis", "PostgreSQL", "Amazon EKS"]),
])
def test_existing_recognition_is_unchanged(sentence, expected):
    assert TOOLS.findall(sentence) == expected
    assert cc.ToolMatcher(TOOLS.base, re.compile(r"(?!)")).findall(sentence) == TOOLS.base.findall(sentence)


@pytest.mark.parametrize("sentence,expected", [
    ("The memorial at Pearl Harbor draws visitors.", []),
    ("Images are pushed to Harbor after the build.", ["Harbor"]),
    ("Lambda was the eleventh letter they studied.", []),
    ("The Lambda function is triggered by the queue.", ["Lambda"]),
    ("The Chef recommended the fish.", []),
    ("Chef cookbooks configure the nodes.", ["Chef"]),
])
def test_ambiguous_names_need_a_related_word_nearby(sentence, expected):
    assert TOOLS.findall(sentence) == expected


@pytest.mark.parametrize("spoken,expected", [
    ("We provision infra with open tofu.", "opentofu"),
    ("Change data lands in click house.", "clickhouse"),
    ("Spend is tracked in kube cost.", "kubecost"),
])
def test_new_spoken_variants_are_corrected(spoken, expected):
    from devops_transcription import clean_transcript
    assert expected in clean_transcript(spoken)


@pytest.mark.parametrize("sentence", [
    "Please open search results in a new tab.",
    "We update the status page during incidents.",
    "There is an open cost question for finance.",
])
def test_ordinary_phrases_are_not_rewritten(sentence):
    from devops_transcription import clean_transcript
    assert clean_transcript(sentence) == sentence


def test_display_names_and_categories_come_from_the_catalog():
    assert _canonicalize_component_term("ec2") == "Amazon EC2"
    assert _canonicalize_component_term("opentofu") == "OpenTofu"
    assert _canonicalize_component_term("Secrets Manager") == "AWS Secrets Manager"
    # Original aliases still take precedence.
    assert _canonicalize_component_term("eks") == "Amazon EKS"
    assert _TECH_CATEGORY_MAP["amazon ec2"] == "Compute"
    assert _TECH_CATEGORY_MAP["kubecost"] == "Cost management"
    # Original categories are not overridden.
    assert _TECH_CATEGORY_MAP["redis"] == "Cache"


def test_catalog_products_reach_the_diagram():
    graph = build_architecture_graph(["Amazon EC2", "Go", "CockroachDB", "Amazon ElastiCache", "Cloudflare"])
    assert graph["central"]["header"] == "Amazon EC2" and graph["central"]["caption"] == "Go"
    assert [n["label"] for n in graph["data"]] == ["CockroachDB", "Amazon ElastiCache"]
    assert graph["entry"][0]["label"] == "Users"  # a CDN is a customer-facing entry point


def test_dynamic_cache_row_does_not_repeat_a_listed_product_under_another_spelling():
    from renderers.sections.system_overview import render
    section = {"id": "system_overview", "title": "Overview", "coverage_content": [], "fields": {
        "key_technologies": {"value": "Amazon ElastiCache, Redis", "source": "cross_section"},
        "cache_layer": {"value": "ElastiCache sits in front of the database.", "source": "semantic"},
    }}
    grid = next(b for b in render(section)["blocks"] if b.get("title") == "Technology summary")
    assert [r["label"] for r in grid["rows"]] == ["Cache"]
