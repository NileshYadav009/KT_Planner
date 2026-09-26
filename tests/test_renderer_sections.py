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
from renderers.sections.first_30_day_ownership import render as render_first_30_day_ownership
from renderers.sections.handover_completion import render as render_handover_completion
from renderers.sections.disaster_recovery import render as render_disaster_recovery
from renderers.sections.ownership_escalation import render as render_ownership_escalation


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


def test_kt_coverage_renders_knowledge_coverage_summary_before_the_matrix():
    # Knowledge coverage (facts identified/mapped/deduplicated/unmapped/lost)
    # is a distinct metric from the Coverage matrix (template-field
    # population) -- must render as its own clearly-labeled block, first,
    # never merged into or confused with the matrix table.
    section = {
        "id": "kt_coverage", "title": "KT Coverage & Knowledge Gaps",
        "_coverage_rows": [{"Domain": "SYSTEM OVERVIEW", "Coverage": "Strong", "Assessment": "5 sentences."}],
        "_knowledge_gaps": [],
        "_knowledge_coverage_summary": {
            "facts_identified": 12, "mapped": 7, "deduplicated": 2, "unmapped": 3, "lost": 0,
        },
    }
    result = render_kt_coverage(section)
    assert result["blocks"][0]["type"] == "NarrativeBlock"
    assert result["blocks"][0]["title"] == "Knowledge coverage"
    text = " ".join(result["blocks"][0]["paragraphs"])
    assert "12" in text and "7 mapped" in text and "2 deduplicated" in text and "3" in text and "0 lost" in text
    assert result["blocks"][1]["type"] == "DecisionTable"


def test_kt_coverage_omits_knowledge_coverage_summary_block_when_absent():
    section = {
        "id": "kt_coverage", "title": "KT Coverage & Knowledge Gaps",
        "_coverage_rows": [{"Domain": "SYSTEM OVERVIEW", "Coverage": "Strong", "Assessment": "5 sentences."}],
        "_knowledge_gaps": [],
    }
    result = render_kt_coverage(section)
    assert not any(b.get("title") == "Knowledge coverage" for b in result["blocks"])


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


def test_open_responsibilities_filters_generic_guidance_out_of_llm_structured_tasks():
    # Regression test for a real bug found via a live Groq-backed pipeline
    # run: the structured extraction prompt explicitly tells the model not
    # to include general safety/escalation guidance as a task, but a
    # smaller/faster model doesn't always follow that instruction --
    # confirmed live: qwen3-8b put a genuine "Open Tasks" transition-plan
    # sentence AND "If you are unsure about an ongoing production activity,
    # contact platform engineering before proceeding." into the SAME
    # "open_tasks" list for one real transcript. The renderer must apply
    # its own guidance-phrase filter to the model's output too, not just
    # the no-LLM fallback path.
    section = {
        "id": "open_responsibilities", "title": "OPEN RESPONSIBILITIES & TRANSITION PLAN",
        "fields": {},
        "coverage_content": [],
        "_structured": {
            "open_tasks": [
                {"task": "Regarding open responsibilities and transition plan, only "
                         "existing and in-progress tasks are handed over. No new initiatives."},
                {"task": "If you are unsure about an ongoing production activity, "
                         "contact platform engineering before proceeding."},
            ],
        },
    }
    result = render_open_responsibilities(section)
    assert not any(b["type"] == "DecisionTable" for b in result["blocks"])
    rendered_text = str(result)
    assert "unsure" not in rendered_text.lower()


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


def test_open_responsibilities_renders_long_unstructured_blob_as_narrative_not_a_fake_table_row():
    # Regression test for a real bug found on a live KT: when structured
    # extraction isn't available, the raw type:"table" field fallback has
    # no way to tell a genuine short task apart from an entire raw
    # transition-plan paragraph -- it produced a single table row whose
    # "Task / Responsibility" cell held a multi-sentence paragraph with
    # every other column blank, which looks like structured data but isn't.
    # A long single-row blob should fall through to the narrative block
    # (sourced from coverage_content) instead.
    long_blob = (
        "Regarding open responsibilities and transition plan, only existing "
        "and in-progress tasks are handed over. No new initiatives are "
        "planned for the incoming owner during the first quarter, and any "
        "additional scope should be raised with the platform engineering "
        "manager before being accepted."
    )
    section = {
        "id": "open_responsibilities", "title": "OPEN RESPONSIBILITIES & TRANSITION PLAN",
        "fields": {"open_tasks": {"value": long_blob}},
        "coverage_content": [long_blob],
    }
    result = render_open_responsibilities(section)
    assert not any(b["type"] == "DecisionTable" for b in result["blocks"])
    assert any(long_blob in str(b) for b in result["blocks"])


def test_open_responsibilities_renders_short_multisentence_blob_as_narrative():
    # The real-world blob that exposed this bug is actually short (17
    # words, under the word-count-only guard's old >20 threshold) because
    # it's two short sentences, not one long one -- confirmed verbatim from
    # a live KT's Open Tasks table.
    real_blob = (
        "Regarding open responsibilities and transition plan, only existing "
        "and in-progress tasks are handed over. No new initiatives."
    )
    section = {
        "id": "open_responsibilities", "title": "OPEN RESPONSIBILITIES & TRANSITION PLAN",
        "fields": {"open_tasks": {"value": real_blob}},
        "coverage_content": [real_blob],
    }
    result = render_open_responsibilities(section)
    assert not any(b["type"] == "DecisionTable" for b in result["blocks"])
    assert any(real_blob in str(b) for b in result["blocks"])


def test_open_responsibilities_renders_generic_guidance_sentence_as_narrative_not_a_fake_task():
    # Regression test for a bug that recurred across nearly every reviewed
    # KT: a short, single-sentence, generic safety/escalation reminder
    # ("If you are unsure about an ongoing production activity, contact
    # platform engineering before proceeding.") isn't caught by a
    # word-count or sentence-count check (it's genuinely one ~13-word
    # sentence) but is explicitly NOT a task -- llm/prompts.py's own
    # open_responsibilities prompt says so by name ("Do NOT include general
    # safety warnings, escalation/contact instructions"). When structured
    # extraction doesn't run, the raw fallback must apply the same
    # judgment via phrasing, not just length.
    guidance = (
        "If you are unsure about an ongoing production activity, "
        "Contact platform engineering before proceeding."
    )
    section = {
        "id": "open_responsibilities", "title": "OPEN RESPONSIBILITIES & TRANSITION PLAN",
        "fields": {"open_tasks": {"value": guidance}},
        "coverage_content": [guidance],
    }
    result = render_open_responsibilities(section)
    assert not any(b["type"] == "DecisionTable" for b in result["blocks"])
    assert any(guidance in str(b) for b in result["blocks"])


def test_open_responsibilities_shows_no_coverage_when_nothing_available():
    section = {"id": "open_responsibilities", "title": "OPEN RESPONSIBILITIES & TRANSITION PLAN", "fields": {}, "coverage_content": []}
    result = render_open_responsibilities(section)
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["type"] != "DecisionTable"


def test_architecture_reference_separates_knowledge_from_metadata():
    # architecture_reference's schema only holds 3 administrative fields
    # (doc link, last updated, verified-by) — real architecture knowledge
    # (the detected component stack) must render as its own "Architecture
    # Knowledge" block, independent of whether those 3 admin fields were
    # ever discussed, and "Architecture Metadata" must always list all 3
    # explicitly rather than silently omitting an undiscussed one.
    section = {
        "id": "architecture_reference", "title": "Architecture Reference",
        "fields": {
            "architecture_link": {"value": "Confluence"},
        },
        "_architecture_components": ["EKS", "React", "CloudFront", "ALB", "FastAPI", "RDS", "Redis", "SQS", "ECR"],
    }
    result = render_architecture_reference(section)
    assert len(result["blocks"]) == 2

    knowledge_block = result["blocks"][0]
    assert knowledge_block["type"] == "NarrativeBlock"
    assert knowledge_block["title"] == "Architecture Knowledge"
    assert "EKS" in knowledge_block["paragraphs"][0]
    assert "SQS" in knowledge_block["paragraphs"][0]

    metadata_block = result["blocks"][1]
    assert metadata_block["type"] == "TechnologyGrid"
    assert metadata_block["title"] == "Architecture Metadata"
    values = {row["label"]: row["value"] for row in metadata_block["rows"]}
    assert values == {
        "Documentation link": "Confluence",
        "Last updated": "Not discussed",
        "Verified by incoming owner": "Not discussed",
    }


def test_architecture_reference_falls_back_to_raw_content_when_no_components_detected():
    # Regression test: a spurious/loosely-related architecture_link value
    # ("Confluence" instead of a real URL) used to be enough on its own to
    # skip the raw coverage_content fallback entirely, collapsing a real
    # 5-bullet Architecture Reference section down to a single
    # "Architecture reference: Confluence" line. Without a detected
    # component list, Architecture Knowledge must fall back to the raw
    # section content instead of disappearing.
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
    assert len(result["blocks"]) == 2
    knowledge_block = result["blocks"][0]
    assert knowledge_block["title"] == "Architecture Knowledge"
    assert len(knowledge_block["paragraphs"]) == 5
    assert "Amazon EKS" in " ".join(knowledge_block["paragraphs"])


def test_architecture_reference_marks_undiscussed_metadata_fields_not_discussed_even_with_no_knowledge():
    section = {"id": "architecture_reference", "title": "Architecture Reference", "fields": {}}
    result = render_architecture_reference(section)
    metadata_block = result["blocks"][-1]
    assert metadata_block["title"] == "Architecture Metadata"
    values = {row["label"]: row["value"] for row in metadata_block["rows"]}
    assert values == {
        "Documentation link": "Not discussed",
        "Last updated": "Not discussed",
        "Verified by incoming owner": "Not discussed",
    }


def test_architecture_reference_renders_details_and_diagram_blocks_when_present():
    section = {
        "id": "architecture_reference", "title": "Architecture Reference",
        "fields": {},
        "_architecture_components": ["React", "CloudFront", "Amazon EKS", "Amazon RDS"],
        "_architecture_sentences": [
            "Amazon RDS PostgreSQL is the primary database.",
            "Redis is used for caching and short-lived session data.",
        ],
        "_architecture_diagram": "Customer\n   │\n   ▼\nReact",
    }
    result = render_architecture_reference(section)
    titles = [b["title"] for b in result["blocks"]]
    assert "Architecture Knowledge" in titles
    assert "Architecture Details" in titles
    assert "High-Level Architecture" in titles
    assert "Architecture Metadata" in titles

    details_block = next(b for b in result["blocks"] if b["title"] == "Architecture Details")
    assert details_block["type"] == "NarrativeBlock"
    assert "Redis is used for caching and short-lived session data." in details_block["paragraphs"]

    diagram_block = next(b for b in result["blocks"] if b["title"] == "High-Level Architecture")
    assert diagram_block["type"] == "CodeBlock"
    assert diagram_block["code"] == "Customer\n   │\n   ▼\nReact"


def test_architecture_reference_omits_details_and_diagram_blocks_when_absent():
    section = {
        "id": "architecture_reference", "title": "Architecture Reference",
        "fields": {},
        "_architecture_components": ["Amazon EKS"],
    }
    result = render_architecture_reference(section)
    titles = [b["title"] for b in result["blocks"]]
    assert "Architecture Details" not in titles
    assert "High-Level Architecture" not in titles


def test_first_30_day_ownership_renders_real_per_week_fields():
    # Regression test: the renderer only ever read coverage_content via a
    # pipe-delimited ("Role | Team") parser, which natural speech never
    # produces — so even when populate_fields() successfully captured all
    # 4 week fields, the rendered table showed one row with a blank second
    # column instead of 4 real rows.
    section = {
        "id": "first_30_day_ownership", "title": "FIRST 30-DAY OWNERSHIP PLAN",
        "fields": {
            "week1": {"label": "Week 1", "value": "Observe, Shadow, and Read Only."},
            "week2": {"label": "Week 2", "value": "Non-prod changes."},
            "week3": {"label": "Week 3", "value": "Prod deployment with supervision."},
            "week4": {"label": "Week 4", "value": "Independent ownership in 4th week."},
        },
        "coverage_content": [],
    }
    result = render_first_30_day_ownership(section)
    assert len(result["blocks"]) == 1
    rows = result["blocks"][0]["rows"]
    assert len(rows) == 4
    assert rows[0] == {"role": "Week 1", "team": "Observe, Shadow, and Read Only."}
    assert all(row["team"] for row in rows)


def test_first_30_day_ownership_falls_back_to_coverage_content_without_fields():
    section = {
        "id": "first_30_day_ownership", "title": "FIRST 30-DAY OWNERSHIP PLAN",
        "fields": {}, "coverage_content": ["Week 1 | Observe and shadow"],
    }
    result = render_first_30_day_ownership(section)
    assert result["blocks"][0]["rows"] == [{"role": "Week 1", "team": "Observe and shadow"}]


# Regression tests for a bug found via a real-world critique of 3 generated
# KT PDFs: when none of the 5 boolean readiness checks (can_deploy,
# understands_rollback, ...) were captured but a "kt_status" closing remark
# was, the old renderer produced ONLY a "KT status: Complete" line with no
# other content in the section at all -- reading as "handover fully done"
# even though none of the substantive readiness checks were confirmed, and
# even when the document's own coverage matrix listed this exact section as
# Missing. The fix always lists all 5 checks (defaulting to "Not covered
# during KT" when absent) and relabels the closing remark so it can't be
# mistaken for a computed completeness verdict.
def test_handover_completion_shows_uncaptured_checks_alongside_closing_remark():
    section = {
        "id": "handover_completion", "title": "HANDOVER COMPLETION CHECK",
        "fields": {"kt_status": {"value": "Complete"}},
        "coverage_content": [],
    }
    result = render_handover_completion(section)
    checklist = next(b for b in result["blocks"] if b["type"] == "ChecklistBlock")
    assert len(checklist["items"]) == 5
    assert all("Not covered during KT" in item for item in checklist["items"])

    narrative = next(b for b in result["blocks"] if b["type"] == "NarrativeBlock")
    joined = " ".join(narrative["paragraphs"])
    assert "KT status:" not in joined
    assert "Closing remark from the KT session: Complete" in joined
    assert "does not by itself confirm" in joined


def test_handover_completion_no_caveat_when_checks_are_actually_confirmed():
    section = {
        "id": "handover_completion", "title": "HANDOVER COMPLETION CHECK",
        "fields": {
            "kt_status": {"value": "Complete"},
            "can_deploy": {"value": True},
            "understands_rollback": {"value": True},
        },
        "coverage_content": [],
    }
    result = render_handover_completion(section)
    narrative = next(b for b in result["blocks"] if b["type"] == "NarrativeBlock")
    joined = " ".join(narrative["paragraphs"])
    assert "does not by itself confirm" not in joined

    checklist = next(b for b in result["blocks"] if b["type"] == "ChecklistBlock")
    assert "Replacement can deploy safely: Confirmed" in checklist["items"]
    assert "Understands rollback: Confirmed" in checklist["items"]
    assert "Knows danger zones: Not covered during KT" in checklist["items"]


def test_handover_completion_falls_back_to_no_coverage_when_nothing_captured():
    section = {
        "id": "handover_completion", "title": "HANDOVER COMPLETION CHECK",
        "fields": {}, "coverage_content": [],
    }
    result = render_handover_completion(section)
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["type"] != "ChecklistBlock" or not result["blocks"][0].get("items")


def test_disaster_recovery_renders_dr_testing_frequency():
    # Regression test for a real bug found auditing a live KT PDF: the
    # structured-extraction JSON schema for disaster_recovery
    # (llm/prompts.py's SECTION_STRUCTURED_PROMPTS) had no field for DR
    # testing cadence at all, so a transcript sentence like "Daily backups
    # are retained for 30 days and DR testing is performed quarterly."
    # silently lost the "quarterly" half — none of rto_steps/rpo_steps/
    # known_failure_scenarios/recovery_contact was a clean fit for it, so
    # the LLM had nowhere to put it. Fixed by adding a dedicated
    # dr_testing_frequency field; this test guards the renderer's half of
    # that fix (the LLM extraction prompt itself isn't unit-testable here).
    section = {
        "id": "disaster_recovery", "title": "DISASTER RECOVERY",
        "fields": {
            "rpo_steps": {"value": "Daily backups retained for 30 days"},
            "dr_testing_frequency": {"value": "quarterly"},
        },
        "coverage_content": [],
    }
    result = render_disaster_recovery(section)
    narrative = next(b for b in result["blocks"] if b["type"] == "NarrativeBlock")
    assert "DR testing frequency: quarterly" in narrative["paragraphs"]

    checklist = next(b for b in result["blocks"] if b["type"] == "ChecklistBlock")
    assert "Daily backups retained for 30 days" in checklist["items"]


def test_disaster_recovery_omits_testing_frequency_line_when_not_captured():
    section = {
        "id": "disaster_recovery", "title": "DISASTER RECOVERY",
        "fields": {"rpo_steps": {"value": "Daily backups retained for 30 days"}},
        "coverage_content": [],
    }
    result = render_disaster_recovery(section)
    assert not any(b["type"] == "NarrativeBlock" for b in result["blocks"])


def test_disaster_recovery_renders_rto_rpo_metric_as_labeled_lines_alongside_procedure():
    # Regression test for a real bug found comparing two live-generated KT
    # PDFs of the same transcript: rto_metric/rpo_metric (field_populator.
    # extract_rto_rpo, a plain duration) used to be written into rto_steps/
    # rpo_steps -- the SAME field ids the LLM's own structured extraction
    # uses for the recovery PROCEDURE narrative -- clobbering it and leaving
    # a bare, unlabeled "2 hours" with no indication of what it measured.
    # Both facts must now coexist: the metric as its own clearly labeled
    # line, the procedure still in the Recovery actions checklist untouched.
    section = {
        "id": "disaster_recovery", "title": "DISASTER RECOVERY",
        "fields": {
            "rto_steps": {"value": "Restore database from Azure SQL backups\nRecreate infrastructure using Bicep"},
            "rto_metric": {"value": "2 hours"},
            "rpo_metric": {"value": "15 minutes"},
        },
        "coverage_content": [],
    }
    result = render_disaster_recovery(section)
    narrative = next(b for b in result["blocks"] if b["type"] == "NarrativeBlock")
    assert "RTO (Recovery Time Objective): 2 hours" in narrative["paragraphs"]
    assert "RPO (Recovery Point Objective): 15 minutes" in narrative["paragraphs"]

    checklist = next(b for b in result["blocks"] if b["type"] == "ChecklistBlock")
    assert "Restore database from Azure SQL backups" in checklist["items"]
    assert "Recreate infrastructure using Bicep" in checklist["items"]
    # The metric must never leak into the procedure checklist as a bare item.
    assert "2 hours" not in checklist["items"]
    assert "15 minutes" not in checklist["items"]


def test_ownership_escalation_separates_guidance_from_oncall_tool_name():
    # Regression test for a real bug found on a live PDF: a general
    # "contact X when unsure" instruction ended up rendered as if it were
    # the on-call tool's NAME ("On-call tool: If you are unsure about a
    # change, involve the appropriate platform or application owner.")
    # because the structured-extraction schema had no field for operational
    # guidance separate from oncall_tool. Fixed with a dedicated field;
    # this guards the renderer keeps them in visually separate blocks.
    section = {
        "id": "ownership_escalation", "title": "OWNERSHIP & ESCALATION",
        "fields": {
            "oncall_tool": {"value": "PagerDuty"},
            "operational_escalation_guidance": {
                "value": "If you are unsure about a change, involve the appropriate platform or application owner."
            },
        },
        "coverage_content": [],
    }
    result = render_ownership_escalation(section)

    ownership_table = next(b for b in result["blocks"] if b["type"] == "OwnershipTable")
    assert {"role": "On-call tool", "team": "PagerDuty"} in ownership_table["rows"]
    assert not any(row["team"].startswith("If you are unsure") for row in ownership_table["rows"])

    guidance_block = next(b for b in result["blocks"] if b["title"] == "Operational escalation guidance")
    assert "If you are unsure about a change" in guidance_block["paragraphs"][0]


def test_danger_zones_splits_a_single_bullet_blob_into_separate_warnings():
    # The LLM polish pass routinely returns a whole section as one
    # "- item. - item." string. That rendered as a single warning card
    # containing both prohibitions plus literal dashes on live output.
    from renderers.sections.common import render_danger_zones
    section = {
        "id": "danger_zones", "title": "DANGER ZONES",
        "coverage_content": [
            "- Production Kubernetes configuration must not be changed manually. "
            "- Do not manually modify Bicep-managed infrastructure without "
            "coordinating with platform engineering."
        ],
    }
    result = render_danger_zones(section)
    warnings = result["blocks"][0]["warnings"]
    assert len(warnings) == 2
    assert warnings[0].startswith("Production Kubernetes configuration")
    assert warnings[1].startswith("Do not manually modify Bicep-managed")
    assert not any(w.startswith("-") for w in warnings)


def test_bullet_split_preserves_hyphenated_words():
    from renderers.blocks.common import split_bullet_blob
    assert split_bullet_blob(["Use the read-only Bicep-managed non-production cluster."]) == [
        "Use the read-only Bicep-managed non-production cluster."
    ]


def test_decision_table_prunes_columns_no_row_has_data_for():
    # A column every row leaves empty renders as nothing but the explicit
    # "Not covered during KT" placeholder repeated down the page -- on live
    # output four of five Day-1 columns were exactly that, squeezing the one
    # column with real content down to an unreadable width.
    from renderers.blocks.table import build_block
    block = build_block(
        "Required access & tools",
        ["Item", "Required", "Location/Link", "Safe on Day-1", "Notes"],
        [{"Item": "Grafana dashboards"}, {"Item": "AKS namespaces"}],
    )
    assert "Item" in block["columns"]
    assert "Location/Link" not in block["columns"]
    assert block["rows"][0]["Item"] == "Grafana dashboards"


def test_decision_table_keeps_partially_filled_columns():
    # A column with SOME data must survive, so a genuine per-row
    # "not covered" signal still shows where it is informative.
    from renderers.blocks.table import build_block
    block = build_block(
        "Failures",
        ["Issue/Symptom", "Likely Cause", "How to Fix"],
        [
            {"Issue/Symptom": "Expired TLS certificate", "Likely Cause": "renewal missed", "How to Fix": ""},
            {"Issue/Symptom": "Service Bus backlog", "Likely Cause": "", "How to Fix": ""},
        ],
    )
    assert block["columns"] == ["Issue/Symptom", "Likely Cause"]


def test_common_failures_renders_first_checks_as_its_own_column():
    # A transcript very often states what to CHECK without stating a fix
    # ("Azure SQL connection exhaustion during high traffic. Check active
    # connections and connection pool metrics."). With no column for it the
    # diagnostic step was dropped entirely -- the schema-gap class of bug.
    section = {
        "id": "common_failures", "title": "COMMON FAILURES",
        "_structured": {"failures": [{
            "symptom": "Azure SQL connection exhaustion",
            "cause": None,
            "first_checks": "Active connections; connection pool metrics",
            "fix": None,
        }]},
    }
    result = render_common_failures(section)
    table = next(b for b in result["blocks"] if b["type"] == "DecisionTable")
    assert "First Checks" in table["columns"]
    assert table["rows"][0]["First Checks"] == "Active connections; connection pool metrics"


def test_common_failures_accepts_first_checks_as_a_list():
    section = {
        "id": "common_failures", "title": "COMMON FAILURES",
        "_structured": {"failures": [{
            "symptom": "Service Bus message backlog",
            "first_checks": ["Consumer pod health", "Processing metrics"],
        }]},
    }
    result = render_common_failures(section)
    table = next(b for b in result["blocks"] if b["type"] == "DecisionTable")
    assert table["rows"][0]["First Checks"] == "Consumer pod health; Processing metrics"


def test_system_overview_categorizes_azure_and_gcp_technologies():
    # _categorize_technologies silently drops any tool with no category, so
    # with an AWS-only map a real Azure KT that identified 18 components
    # rendered a Technology summary of just three rows.
    section = {
        "id": "system_overview", "title": "SYSTEM OVERVIEW",
        "fields": {"key_technologies": {"value": (
            "Angular, Azure Kubernetes Service, Azure SQL, Azure Service Bus, "
            "Azure Front Door, Bicep, Azure Key Vault, Azure Monitor, .NET"
        )}},
        "coverage_content": [],
    }
    result = render_system_overview(section)
    grid = next(b for b in result["blocks"] if b.get("title") == "Technology summary")
    labels = {row["label"] for row in grid["rows"]}
    for expected in ("Frontend", "Compute", "Database", "Messaging", "Edge / ingress",
                     "Infrastructure", "Secrets", "Observability", "Backend"):
        assert expected in labels, f"missing category {expected}: {labels}"


def test_environments_surfaces_content_not_covered_by_a_table_row():
    # The table has one row per schema-declared environment, so a transcript
    # naming any OTHER environment had nowhere to go -- "The platform has
    # development, QA, staging, and production environments." was dropped
    # outright on a live run, losing the dev and QA environments entirely.
    section = {
        "id": "environments", "title": "ENVIRONMENTS",
        "fields": {"staging_notes": {"value": "Staging has mocked payment integrations."}},
        "coverage_content": [
            "Staging has mocked payment integrations.",
            "The platform has development, QA, staging, and production environments.",
        ],
    }
    result = render_environments(section)
    extra = next(b for b in result["blocks"] if b.get("title") == "Additional environment notes")
    assert extra["paragraphs"] == [
        "The platform has development, QA, staging, and production environments."
    ]
    # The sentence already shown as a table row is not repeated.
    assert not any("mocked payment" in p for p in extra["paragraphs"])


def test_environments_omits_extra_notes_block_when_rows_cover_everything():
    section = {
        "id": "environments", "title": "ENVIRONMENTS",
        "fields": {"staging_notes": {"value": "Staging has mocked payment integrations."}},
        "coverage_content": ["Staging has mocked payment integrations."],
    }
    result = render_environments(section)
    assert not any(b.get("title") == "Additional environment notes" for b in result["blocks"])


def test_technology_summary_drops_a_generic_term_when_the_branded_one_is_present():
    # A transcript says the full product name once and the generic word
    # afterwards ("Azure Kubernetes Service ... Kubernetes workloads"), and
    # both match the tools regex -- the summary then listed the same tier
    # twice as "Azure Kubernetes Service; Kubernetes".
    section = {
        "id": "system_overview", "title": "SYSTEM OVERVIEW",
        "fields": {"key_technologies": {"value": (
            "Azure Kubernetes Service, Kubernetes, Azure Service Bus, Service Bus, Redis"
        )}},
        "coverage_content": [],
    }
    result = render_system_overview(section)
    grid = next(b for b in result["blocks"] if b.get("title") == "Technology summary")
    by_label = {row["label"]: row["value"] for row in grid["rows"]}
    assert by_label["Compute"] == "Azure Kubernetes Service"
    assert by_label["Messaging"] == "Azure Service Bus"
    # A term with no branded superset is untouched.
    assert by_label["Cache"] == "Redis"


def test_day1_access_items_render_as_a_checklist_not_a_contradictory_table():
    # Natural speech gives the ITEMS only, so every other column filled with
    # "Not covered during KT" -- and the table keeps a minimum of two
    # columns, so a real PDF printed "Grafana dashboards | Not covered
    # during KT", which contradicts itself: the item is listed precisely
    # BECAUSE the transcript named it as required. See §9rr.
    from renderers.sections.day1 import render

    out = render({
        "id": "day1_survival_checklist",
        "title": "DAY-1 SURVIVAL CHECKLIST",
        "fields": {
            "required_access": {
                "value": "Grafana dashboards\nAzure DevOps repositories\nAKS namespaces"
            }
        },
    })

    access = next(b for b in out["blocks"] if b.get("title") == "Required access & tools")
    assert access["type"] == "ChecklistBlock"
    assert access["items"] == [
        "Grafana dashboards",
        "Azure DevOps repositories",
        "AKS namespaces",
    ]
    rendered = " ".join(access["items"])
    assert "Not covered" not in rendered


def test_day1_still_renders_a_table_when_columns_have_real_data():
    # The checklist path must not swallow genuinely tabular evidence.
    from renderers.sections.day1 import render

    out = render({
        "id": "day1_survival_checklist",
        "title": "DAY-1 SURVIVAL CHECKLIST",
        "fields": {
            "required_access": {"value": "Grafana | Yes | https://grafana.internal"}
        },
    })

    access = next(b for b in out["blocks"] if b.get("title") == "Required access & tools")
    assert access["type"] != "ChecklistBlock"


def test_technology_summary_categorises_a_full_aws_stack():
    # A recognised tool with no category is silently dropped by
    # _categorize_technologies(), so vocabulary and category map must move
    # together. This is the AWS half of a defect already fixed for Azure.
    from renderers.sections.system_overview import _categorize_technologies

    rows = _categorize_technologies(
        "React, Amazon EKS, Spring Boot, Aurora PostgreSQL, Redis, Amazon MSK, "
        "OpenSearch, Route 53, AWS WAF, AWS KMS, Amazon ECR, Terraform"
    )
    by_label = {r["label"]: r["value"] for r in rows}

    assert "Aurora PostgreSQL" in by_label["Database"]
    assert "Amazon MSK" in by_label["Messaging"]
    assert "OpenSearch" in by_label["Observability"]
    assert "Route 53" in by_label["Edge / ingress"]
    assert "Spring Boot" in by_label["Backend"]
    assert "AWS KMS" in by_label["Security"]


def test_failure_table_keeps_the_triggering_condition():
    # issue -> condition -> cause -> first checks -> fix must stay intact.
    # "Azure SQL connection exhaustion during high traffic" previously
    # rendered without "during high traffic" -- there was no column for it,
    # so the condition was dropped. See §9ss.
    from renderers.sections.common_failures import render, STRUCTURED_COLUMNS

    assert "When it happens" in STRUCTURED_COLUMNS

    out = render({
        "id": "common_failures",
        "title": "COMMON FAILURES & FIXES",
        "_structured": {"failures": [{
            "symptom": "Azure SQL connection exhaustion",
            "condition": "During high traffic",
            "first_checks": ["Active Connections", "Connection Pool metrics"],
            "cause": None,
            "fix": None,
        }]},
    })
    row = next(b for b in out["blocks"] if b.get("rows"))["rows"][0]

    assert row["Issue/Symptom"] == "Azure SQL connection exhaustion"
    assert row["When it happens"] == "During high traffic"
    assert "Active Connections" in row["First Checks"]
