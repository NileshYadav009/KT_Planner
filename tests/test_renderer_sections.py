"""Tests for the renderer rewrites/additions made to match the golden-
reference KT structure: system_overview.py's field-id fix + technology
summary, the new environments.py renderer, common_failures.py's historical
incident sub-block, and known_bad_days.py's Operational Calendar cost-
pattern table.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from renderers.sections.system_overview import render as render_system_overview
from renderers.sections.environments import render as render_environments
from renderers.sections.common_failures import render as render_common_failures
from renderers.sections.known_bad_days import render as render_known_bad_days
from renderers.sections.kt_coverage import render as render_kt_coverage


def test_system_overview_renders_real_schema_fields_not_stale_ones():
    # Regression test: the renderer used to check field ids (cache_layer,
    # event_streaming, documentation_links, system_description) that don't
    # exist anywhere in kt_schema_new.json, so only business_criticality
    # ever rendered. Must now read the real fields.
    section = {
        "id": "system_overview", "title": "SYSTEM OVERVIEW",
        "fields": {
            "business_criticality": {"value": "High"},
            "orders_per_day": {"value": "50,000 orders per day"},
            "impact_if_down": {"what_breaks": {"value": "Customers cannot place orders"}},
        },
        "coverage_content": [],
    }
    result = render_system_overview(section)
    grid = next(b for b in result["blocks"] if b["title"] == "Captured knowledge")
    labels = {row["label"] for row in grid["rows"]}
    assert labels == {"Business criticality", "Business volume", "Customer impact"}


def test_system_overview_technology_summary_categorizes_known_tools():
    section = {
        "id": "system_overview", "title": "SYSTEM OVERVIEW",
        "fields": {"key_technologies": {"value": "React, PostgreSQL, Terraform, PagerDuty"}},
        "coverage_content": [],
    }
    result = render_system_overview(section)
    grid = next(b for b in result["blocks"] if b["title"] == "Technology summary")
    by_label = {row["label"]: row["value"] for row in grid["rows"]}
    assert by_label["Frontend"] == "React"
    assert by_label["Database"] == "PostgreSQL"
    assert by_label["Infrastructure"] == "Terraform"
    assert by_label["Alerting"] == "PagerDuty"


def test_system_overview_falls_back_to_coverage_content_when_no_fields_populated():
    section = {
        "id": "system_overview", "title": "SYSTEM OVERVIEW",
        "fields": {},
        "coverage_content": ["Some free-form overview text."],
    }
    result = render_system_overview(section)
    assert result["blocks"][0]["type"] == "NarrativeBlock"
    assert "Some free-form overview text." in result["blocks"][0]["paragraphs"]


def test_environments_renders_populated_environments_as_table():
    section = {
        "id": "environments", "title": "ENVIRONMENTS",
        "fields": {
            "staging_notes": {"value": "Closely mirrors production; payments are mocked."},
        },
        "coverage_content": [],
    }
    result = render_environments(section)
    table = next(b for b in result["blocks"] if b["type"] == "DecisionTable")
    assert table["rows"] == [{
        "Environment": "Staging",
        "Known characteristics": "Closely mirrors production; payments are mocked.",
    }]
    # Must always include the "do not over-infer" callout when it renders a table.
    assert any(b["title"] == "Do not over-infer" for b in result["blocks"])


def test_environments_no_coverage_when_nothing_populated():
    section = {"id": "environments", "title": "ENVIRONMENTS", "fields": {}, "coverage_content": []}
    result = render_environments(section)
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["type"] == "NarrativeBlock"


def test_common_failures_adds_historical_incident_block_for_past_occurrence():
    section = {
        "id": "common_failures", "title": "COMMON FAILURES & FIXES",
        "_structured": {"failures": [
            {"symptom": "Database connection pool exhaustion", "cause": "Flash sales", "fix": "", "frequency": "common"},
            {"symptom": "Major outage", "cause": "Redis memory saturation", "when": "Last year", "impact": "Major outage"},
        ]},
    }
    result = render_common_failures(section)
    assert len(result["blocks"]) == 2
    historical = result["blocks"][1]
    assert historical["title"] == "Historical incident record"
    row = historical["rows"][0]
    assert row["Incident"] == "Major outage"
    assert row["When"] == "Last year"
    assert row["Resolution"] == "Not covered in KT"


def test_common_failures_no_historical_block_when_nothing_has_a_when():
    section = {
        "id": "common_failures", "title": "COMMON FAILURES & FIXES",
        "_structured": {"failures": [{"symptom": "Pod crash", "cause": "Missing secret", "fix": "Verify secrets"}]},
    }
    result = render_common_failures(section)
    assert len(result["blocks"]) == 1


def test_known_bad_days_renders_operational_calendar_with_cost_patterns():
    section = {
        "id": "known_bad_days", "title": "OPERATIONAL CALENDAR",
        "coverage_content": ["Avoid deployments during Black Friday."],
        "_cost_patterns": [{"label": "Spot instances", "value": "Non-production environment"}],
    }
    result = render_known_bad_days(section)
    assert len(result["blocks"]) == 2
    assert result["blocks"][0]["type"] == "WarningBlock"
    assert result["blocks"][1]["type"] == "TechnologyGrid"
    assert result["blocks"][1]["title"] == "Cost-related operating patterns"


def test_kt_coverage_renders_knowledge_gaps_as_distinct_checklist():
    section = {
        "id": "kt_coverage", "title": "KT Coverage & Knowledge Gaps",
        "_coverage_rows": [{"Domain": "SIGN-OFF", "Coverage": "Missing", "Assessment": "Not covered in the KT session."}],
        "_knowledge_gaps": ["SIGN-OFF"],
    }
    result = render_kt_coverage(section)
    assert len(result["blocks"]) == 3
    assert result["blocks"][0]["type"] == "DecisionTable"
    assert result["blocks"][1]["type"] == "NarrativeBlock"
    gaps_block = result["blocks"][2]
    assert gaps_block["type"] == "ChecklistBlock"
    assert gaps_block["items"] == ["SIGN-OFF"]


def test_kt_coverage_omits_gaps_block_when_nothing_missing():
    section = {
        "id": "kt_coverage", "title": "KT Coverage & Knowledge Gaps",
        "_coverage_rows": [{"Domain": "SYSTEM OVERVIEW", "Coverage": "Strong", "Assessment": "5 sentences."}],
        "_knowledge_gaps": [],
    }
    result = render_kt_coverage(section)
    assert len(result["blocks"]) == 2
    assert not any(b["type"] == "ChecklistBlock" for b in result["blocks"])
