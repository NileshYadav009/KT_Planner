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
from renderers.sections.open_responsibilities import render as render_open_responsibilities
from renderers.sections.architecture_reference import render as render_architecture_reference


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


def test_open_responsibilities_prefers_structured_rows_over_raw_table_fallback():
    # llm/prompts.py's open_responsibilities structured prompt is instructed
    # to only emit rows for genuine tasks/recurring duties and drop general
    # safety/escalation guidance — when it produced real rows, those must be
    # used instead of the raw type:"table" fallback (which has no way to
    # tell a stray sentence apart from a real task).
    section = {
        "id": "open_responsibilities", "title": "OPEN RESPONSIBILITIES & TRANSITION PLAN",
        "fields": {
            "open_tasks": {"value": "If you are unaware about my production activity, contact platform engineering before proceeding."},
        },
        "coverage_content": [],
        "_structured": {
            "open_tasks": [
                {"task": "Finish migrating the batch job to EKS", "type": "Project", "status": "In progress"},
            ],
            "recurring_responsibilities": [
                {"activity": "Rotate Vault secrets", "frequency": "Quarterly", "owner_before": "Outgoing owner"},
            ],
        },
    }
    result = render_open_responsibilities(section)
    assert len(result["blocks"]) == 2
    tasks_block, recurring_block = result["blocks"]
    assert tasks_block["title"] == "Open tasks"
    assert tasks_block["rows"] == [{
        "Task / Responsibility": "Finish migrating the batch job to EKS",
        "Type": "Project", "Current Status": "In progress", "Business Impact": "",
        "Knowledge Transfer Done": "", "Recommendation": "", "Incoming Owner Decision": "",
    }]
    assert recurring_block["title"] == "Recurring responsibilities"
    # The safety-guidance sentence in fields["open_tasks"] must not appear
    # anywhere in the structured-preferred output.
    rendered_text = str(result)
    assert "production activity" not in rendered_text


def test_open_responsibilities_falls_back_to_raw_table_without_structured_data():
    section = {
        "id": "open_responsibilities", "title": "OPEN RESPONSIBILITIES & TRANSITION PLAN",
        "fields": {
            "open_tasks": {"value": "Finish migrating the batch job to EKS"},
        },
        "coverage_content": [],
    }
    result = render_open_responsibilities(section)
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["type"] == "DecisionTable"
    assert result["blocks"][0]["rows"][0]["Task / Responsibility"] == "Finish migrating the batch job to EKS"


def test_open_responsibilities_shows_no_coverage_when_nothing_available():
    section = {"id": "open_responsibilities", "title": "OPEN RESPONSIBILITIES & TRANSITION PLAN", "fields": {}, "coverage_content": []}
    result = render_open_responsibilities(section)
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["type"] != "DecisionTable"


def test_architecture_reference_does_not_let_one_thin_field_hide_a_richer_fallback():
    # Regression test: a spurious/loosely-related architecture_link value
    # ("Confluence" instead of a real URL) used to be enough on its own to
    # skip the raw coverage_content fallback entirely, collapsing a real
    # 5-bullet Architecture Reference section down to a single
    # "Architecture reference: Confluence" line.
    section = {
        "id": "architecture_reference", "title": "Architecture Reference",
        "fields": {
            "architecture_link": {"value": "Confluence"},
        },
        "coverage_content": [
            "All services run on Amazon EKS.",
            "Traffic enters through CloudFront and the application load balancer before reaching the Kubernetes workload.",
            "One tribal knowledge item is that CloudFront cache invalidation can take longer than expected during large releases.",
            "The primary database runs on Amazon RDS with PostgreSQL, with multiple AZs enabled.",
            "The architecture diagram is maintained in Confluence and should be reviewed before making infrastructure changes.",
        ],
    }
    result = render_architecture_reference(section)
    assert len(result["blocks"]) == 1
    block = result["blocks"][0]
    assert block["type"] == "NarrativeBlock"
    assert len(block["paragraphs"]) == 5
    assert "Amazon EKS" in " ".join(block["paragraphs"])


def test_architecture_reference_prefers_real_field_content_when_richer_than_fallback():
    # When the fields genuinely captured more than the (thin/absent)
    # fallback, they should still be used — this isn't "always prefer the
    # fallback", only "don't let a thin field hide a richer fallback".
    section = {
        "id": "architecture_reference", "title": "Architecture Reference",
        "fields": {
            "architecture_link": {"value": "https://confluence.example.com/architecture"},
        },
        "coverage_content": ["See the docs."],
    }
    result = render_architecture_reference(section)
    assert len(result["blocks"]) == 1
    assert "https://confluence.example.com/architecture" in result["blocks"][0]["paragraphs"][0]
