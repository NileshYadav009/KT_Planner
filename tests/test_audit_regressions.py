"""Regression tests for the end-to-end mapping audit (docs/REPOSITORY_AUDIT.md
§9uu). Each test pins one defect found by tracing a real ~2,900-word KT
(transcript -> cleaning -> segmentation -> classification -> field
population -> rendering -> PDF) and comparing every transcript sentence with
the final document. All are LLM-free and model-free, so they run fast.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from architecture_diagram import build_architecture_flow_diagram
from context_mapper import (
    AudioSegment,
    Classification,
    Sentence,
    _is_faithful_repair,
    segment_sentences,
)
from devops_transcription import clean_transcript
from field_populator import (
    _extract_by_pattern,
    extract_business_volume,
    extract_customer_reach,
    extract_rto_rpo,
    populate_fields,
)
from knowledge.knowledge_builder import (
    _canonicalize_component_term,
    _drop_subsumed_components,
    verify_document_coverage,
)
from pdf_rendering import append_residual_content
from renderers.sections.environments import render as render_environments
from renderers.sections.ownership_escalation import render as render_ownership
from renderers.sections.security import render as render_security
from renderers.sections.signoff import render as render_signoff
from section_rules import apply_continuation_rules, match_section_rules


# --------------------------------------------------------------------------
# Transcript cleaning must not change meaning
# --------------------------------------------------------------------------

def test_cleaning_keeps_content_words_that_are_also_fillers():
    text = clean_transcript(
        "Cost optimization uses right-sizing and production-like staging. "
        "Review the matrix so that they understand ownership. It works well."
    )
    assert "right-sizing" in text
    assert "production-like" in text
    assert "so that they understand" in text
    assert "works well" in text


def test_cleaning_still_removes_discourse_fillers():
    text = clean_transcript("So, um, the database is, you know, Aurora. Okay so we use Kafka, right?")
    assert text == "The database is Aurora. We use Kafka."


def test_cleaning_does_not_delete_or_invert_ordinary_words():
    text = clean_transcript(
        "A healthy canary allows the deployment to continue. The migration is done. "
        "The config was redone after formal sign-off."
    )
    assert "deployment to continue." in text
    assert "The migration is done" in text  # was rewritten to "goes down"
    assert "redone" in text and "red zone" not in text
    assert "formal sign-off" in text  # "ter form" rule used to fire inside "after formal"


def test_cleaning_keeps_acronym_casing_and_inflections():
    text = clean_transcript(
        "The platform runs in AWS. Route 53 handles DNS. "
        "Terraform manages load balancers and canary deployments that can be rolled back."
    )
    assert "AWS" in text and " aws" not in text
    assert "Route 53" in text
    assert "load balancers" in text
    assert "canary deployments" in text
    assert "rolled back" in text


def test_cleaning_still_corrects_lowercase_garbles():
    text = clean_transcript("pager duty is used for alerting and rabbit mq handles messaging.")
    assert "PagerDuty" in text and "RabbitMQ" in text


# --------------------------------------------------------------------------
# Segmentation must keep every clause of a long sentence
# --------------------------------------------------------------------------

def test_long_sentence_split_keeps_every_clause():
    long_sentence = (
        "The customer applications are used by institutional and retail customers across "
        "North America, Europe, and Asia-Pacific, and because this is a financial trading platform, "
        "availability and ordering correctness are more important than simply maximizing throughput, "
        "which is why the platform is designed around strict ordering guarantees."
    )
    assert len(long_sentence) > 240
    sentences = segment_sentences([AudioSegment(text=long_sentence, start=0, end=0, avg_logprob=-0.3)])
    joined = " ".join(s.text for s in sentences)
    for clause in ("institutional and retail customers", "North America, Europe", "ordering correctness", "strict ordering"):
        assert clause in joined, clause


# --------------------------------------------------------------------------
# Section routing
# --------------------------------------------------------------------------

def test_a_role_mention_alone_does_not_route_to_ownership():
    rule = match_section_rules(
        "However, not every service has automated rollback, so the on-call engineer still needs "
        "to understand which services have automated versus manual rollback behavior."
    )
    assert rule.section_id == "deployment_and_rollback"
    rule = match_section_rules("If the on-call engineer cannot resolve the issue, escalation goes to the platform engineering manager.")
    assert rule.section_id == "ownership_escalation"


def test_deployment_freeze_routes_to_operational_calendar():
    rule = match_section_rules(
        "Production deployments are frozen during these periods unless the platform engineering manager approves."
    )
    assert rule.section_id == "known_bad_days"


def test_singular_new_team_member_routes_to_day1():
    rule = match_section_rules("For a new team member, the first day should include access to Grafana, GitHub and ArgoCD.")
    assert rule.section_id == "day1_survival_checklist"


def test_referring_to_danger_zones_is_not_a_danger_zone():
    rule = match_section_rules(
        "The handover is considered complete only when the incoming engineer has reviewed the danger zones."
    )
    assert rule.section_id == "handover_completion"


def _classified(text, section_id):
    sentence = Sentence(text=text, start=0.0, end=0.0)
    primary = Classification(section_id=section_id, section_title=section_id, confidence=0.6, reason="", similarity_score=0.6)
    return SimpleNamespace(
        sentence=sentence, primary_classification=primary, is_unassigned=False, explainability_log=None,
    )


def test_enumerated_items_follow_their_list_heading_not_each_other():
    sentences = [
        _classified("There are currently several open responsibilities.", "open_responsibilities"),
        _classified("The first is to review the certificate renewal configuration.", "handover_completion"),
        _classified("The second is to document the Kafka consumer-lag procedure.", "known_bad_days"),
        _classified("The third is to review which services have automated rollback.", "deployment_and_rollback"),
    ]
    apply_continuation_rules(sentences, {})
    assert [s.primary_classification.section_id for s in sentences] == ["open_responsibilities"] * 4


def test_remediation_and_fragments_follow_the_sentence_they_continue():
    sentences = [
        _classified("One common problem is Aurora connection exhaustion.", "common_failures"),
        _classified("The usual remediation is to correct the connection pool configuration.", "disaster_recovery"),
        _classified("Do not delete Kafka topics, modify Terraform state,", "danger_zones"),
        _classified("or bypass the production deployment approval process.", "handover_completion"),
    ]
    sentences[3].sentence.raw_text = "or bypass the production deployment approval process."
    apply_continuation_rules(sentences, {})
    assert sentences[1].primary_classification.section_id == "common_failures"
    assert sentences[3].primary_classification.section_id == "danger_zones"


# --------------------------------------------------------------------------
# Field population: no invented values, no forced fits
# --------------------------------------------------------------------------

def _boolean(field_id, label=""):
    return {"id": field_id, "label": label, "type": "boolean"}


def test_readiness_checks_need_topic_evidence_and_respect_negation():
    text = (
        "The staging deployment exercise has been completed successfully.\n"
        "The staging rollback exercise has also been completed successfully.\n"
        "Formal sign-off has not yet been submitted because the incoming owner still needs "
        "to complete the canary rollback review."
    )
    assert _extract_by_pattern(_boolean("can_deploy", "Replacement can deploy safely"), text) is True
    # Positive AND outstanding evidence about rollback -> not confirmed.
    assert _extract_by_pattern(_boolean("understands_rollback"), text) is False
    # Nothing about danger zones -> no claim either way.
    assert _extract_by_pattern(_boolean("knows_danger_zones"), text) is None


def test_select_options_need_a_topic_and_a_whole_word():
    kt_status = {"id": "kt_status", "label": "", "type": "single_select", "options": ["Complete", "Needs Follow-up"]}
    assert _extract_by_pattern(kt_status, "The staging deployment exercise has been completed successfully.") is None
    users = {"id": "users", "label": "Who uses it?", "type": "multi_select", "options": ["B2B", "B2C", "Internal"]}
    assert _extract_by_pattern(users, "Some internal services are written in Go.") is None
    assert _extract_by_pattern(users, "The system is used internally by the fulfillment team.") == ["Internal"]


def test_explicit_criticality_statement_is_captured_without_an_llm():
    field = {"id": "business_criticality", "label": "Business Criticality", "type": "single_select", "options": ["High", "Medium", "Low"]}
    assert _extract_by_pattern(field, "This is one of the most critical platforms operated by the organization.") == "High"


def test_spelled_out_numbers_are_captured():
    assert extract_rto_rpo("The recovery objective is an RTO of two hours and an RPO of fifteen minutes.") == {
        "rto_metric": "two hours", "rpo_metric": "fifteen minutes",
    }
    volume = extract_business_volume(
        "On a normal business day the platform processes between eight and twelve million order events."
    )
    assert volume and "twelve million order events" in volume
    rollback = {"id": "rollback_time", "label": "Expected rollback time", "type": "text"}
    assert _extract_by_pattern(rollback, "The target for a normal application rollback is fifteen minutes.") == "fifteen minutes"


def test_customer_reach_keeps_proper_noun_casing():
    reach = extract_customer_reach("Used by institutional and retail customers across North America, Europe, and Asia-Pacific.")
    assert "institutional and retail customers" in reach
    assert "North America, Europe, and Asia-Pacific" in reach


def test_documentation_location_fills_a_link_field_when_no_url_is_spoken():
    field = {"id": "architecture_link", "label": "Link to detailed architecture documentation", "type": "url"}
    value = _extract_by_pattern(field, "The architecture documentation is maintained in the internal engineering wiki.")
    assert value and "wiki" in value


def test_name_fields_are_never_filled_with_a_similar_sentence():
    schema = [{"id": "signoff", "title": "Sign-off", "fields": [
        {"id": "outgoing_owner", "label": "", "type": "text"},
        {"id": "incoming_owner", "label": "", "type": "text"},
    ]}]
    section_content = {"signoff": {"sentences": [
        {"text": "The outgoing owner has therefore not marked the handover as fully closed."},
    ]}}
    result = populate_fields(schema, {"signoff": {}}, llm_provider=None, embedding_model=None, section_content=section_content)
    assert result["signoff"]["outgoing_owner"]["source"] == "unfilled"
    assert result["signoff"]["incoming_owner"]["source"] == "unfilled"


def test_specific_field_claims_its_sentence_before_a_catch_all_table():
    schema = [{"id": "deployment_and_rollback", "title": "Deployment", "fields": [
        {"id": "deployment_steps", "label": "Normal Deployment Process", "type": "table"},
        {"id": "deployment_window", "label": "Deployment Window", "type": "text"},
    ]}]
    section_content = {"deployment_and_rollback": {"sentences": [
        {"text": "The deployment process starts when a pull request is merged."},
        {"text": "The normal production deployment window is Tuesday between 20:00 and 22:00 UTC."},
    ]}}
    result = populate_fields(schema, {}, llm_provider=None, embedding_model=None, section_content=section_content)
    fields = result["deployment_and_rollback"]
    assert "Tuesday" in fields["deployment_window"]["value"]
    assert "Tuesday" not in fields["deployment_steps"]["value"]


# --------------------------------------------------------------------------
# Rendering: every classified sentence reaches the document
# --------------------------------------------------------------------------

def test_residual_content_is_appended_when_a_renderer_drops_it():
    section = {
        "id": "security_controls", "title": "Security",
        "fields": {"security_scan_config": {"value": "Security controls include Trivy container scanning."}},
        "coverage_content": [
            "Security controls include Trivy container scanning.",
            "For secrets, the platform uses HashiCorp Vault.",
            "Container images are stored in Amazon ECR.",
        ],
    }
    rendered = append_residual_content(section, render_security(section))
    residual = [b for b in rendered["blocks"] if b.get("title") == "Additional details from the KT session"]
    assert residual, rendered
    assert residual[0]["paragraphs"] == [
        "Security controls include Trivy container scanning.",
        "For secrets, the platform uses HashiCorp Vault.",
        "Container images are stored in Amazon ECR.",
    ]
    # The grid holds tool names, not the sentence.
    grid = next(b for b in rendered["blocks"] if b["type"] == "TechnologyGrid")
    assert grid["rows"][0]["value"] == "Trivy"


def test_residual_does_not_duplicate_what_a_table_row_already_shows():
    section = {
        "id": "ownership_escalation", "title": "Ownership",
        "fields": {},
        "coverage_content": ["The market connectivity team owns external market-data connectivity and provider-side configuration."],
    }
    rendered = append_residual_content(section, render_ownership(section))
    table = next(b for b in rendered["blocks"] if b["type"] == "OwnershipTable")
    assert table["rows"] == [{"role": "External market-data connectivity and provider-side configuration", "team": "Market connectivity team"}]
    assert not any(b.get("title") == "Additional details from the KT session" for b in rendered["blocks"])


def test_unfilled_approval_is_not_rendered_as_not_yet():
    rendered = render_signoff({"id": "signoff", "title": "Sign-off", "fields": {
        "outgoing_owner": {"value": "Priya"}, "approved": {"value": "", "source": "unfilled"},
    }})
    labels = [r["label"] for b in rendered["blocks"] for r in b.get("rows", [])]
    assert "Approved by incoming owner" not in labels


def test_environment_note_only_lists_details_that_were_not_mentioned():
    rendered = render_environments({"id": "environments", "title": "Environments", "fields": {}, "coverage_content": [
        "The environments are development, integration, staging, pre-production, and production.",
    ]})
    note = next(b for b in rendered["blocks"] if b["title"] == "Do not over-infer")["paragraphs"][0]
    assert "environment names" not in note
    assert "URLs" in note


def test_document_coverage_check_recovers_sentences_missing_from_the_document():
    ko = {
        "sections": [{"id": "kt_coverage", "title": "KT Coverage", "_coverage_rows": [{"Domain": "X", "Coverage": "Strong", "Assessment": "."}]}],
        "rendered_sections": [
            {"section_id": "monitoring_observability", "section_title": "Monitoring", "blocks": [
                {"type": "NarrativeBlock", "title": "Monitoring", "paragraphs": ["Grafana shows request rate and latency."]},
            ]},
            {"section_id": "kt_coverage", "section_title": "KT Coverage", "blocks": []},
        ],
    }
    result = verify_document_coverage(ko, [
        "Grafana shows request rate and latency.",
        "Aurora backups are retained for thirty five days in the recovery region.",
    ])
    summary = result["_knowledge_coverage_summary"]
    assert summary == {"facts_identified": 2, "mapped": 1, "unmapped": 0, "recovered": 1, "lost": 0, "verified_against_document": True}
    notes = next(s for s in result["rendered_sections"] if s["section_id"] == "unmapped_findings")
    assert "Aurora backups are retained" in notes["blocks"][0]["paragraphs"][0]


# --------------------------------------------------------------------------
# Architecture representation
# --------------------------------------------------------------------------

def test_component_aliases_and_vendor_prefixed_duplicates_collapse():
    names = [_canonicalize_component_term(n) for n in ["EKS", "Amazon EKS", "ALB", "Application Load Balancer", "load balancer", "WAF", "AWS WAF", "Aurora", "Aurora PostgreSQL", "Vault", "Azure Key Vault"]]
    kept = _drop_subsumed_components(list(dict.fromkeys(names)))
    assert kept == ["Amazon EKS", "Application Load Balancer", "AWS WAF", "Aurora PostgreSQL", "Vault", "Azure Key Vault"]


def test_diagram_draws_every_dependency_and_the_stated_request_path():
    diagram = build_architecture_flow_diagram([
        "Route 53", "CloudFront", "AWS WAF", "Application Load Balancer", "Amazon EKS", "Spring Boot",
        "Aurora PostgreSQL", "Redis", "Kafka", "Amazon MSK", "Amazon SQS", "Vault", "AWS KMS",
    ])
    chain = [line for line in diagram.splitlines() if line in ("Customer", "Route 53", "CloudFront", "AWS WAF", "Application Load Balancer", "Amazon EKS")]
    assert chain[:6] == ["Customer", "Route 53", "CloudFront", "AWS WAF", "Application Load Balancer", "Amazon EKS"]
    for dependency in ("Aurora PostgreSQL", "Redis", "Kafka (Amazon MSK)", "Amazon SQS"):
        assert f"► {dependency}" in diagram, dependency
    assert "Vault, AWS KMS ──► Secrets & keys" in diagram


# --------------------------------------------------------------------------
# LLM sentence repair must not replace transcript text with a free answer
# --------------------------------------------------------------------------

def test_llm_repair_output_must_be_a_faithful_correction():
    assert _is_faithful_repair("the pods restart when redis is full", "The pods restart when Redis is full.", "")
    assert not _is_faithful_repair(
        "the pods restart",
        "Sure! Here is a detailed explanation of Kubernetes pod lifecycle management and restart policies.",
        "",
    )


def test_a_stated_requirement_does_not_confirm_a_readiness_check():
    text = (
        "The handover is considered complete only when the incoming engineer has reviewed "
        "the danger zones and understands the escalation path."
    )
    assert _extract_by_pattern(_boolean("knows_danger_zones"), text) is None
    assert _extract_by_pattern(_boolean("escalation_clear"), text) is None


def test_short_sentence_is_not_hidden_by_a_longer_one_sharing_most_words():
    from pdf_rendering import is_text_represented, _norm_for_match
    rendered = [_norm_for_match(
        "The handover is complete only when the engineer has successfully completed the staging deployment and rollback."
    )]
    assert not is_text_represented("The staging deployment exercise has been completed successfully.", rendered)


def test_chunk_rendered_as_separate_rows_is_not_reported_missing():
    from pdf_rendering import is_text_represented, _norm_for_match
    rendered = [_norm_for_match(s) for s in (
        "EKS infrastructure and Terraform Platform engineering team",
        "Aurora configuration and database performance Database engineering team",
    )]
    chunk = ("The platform engineering team owns EKS infrastructure and Terraform. "
             "The database engineering team owns Aurora configuration and database performance.")
    assert is_text_represented(chunk, rendered)


def test_paraphrased_fact_in_its_own_section_is_not_recovered_as_missing():
    sentence = ("During the first thirty days, the incoming engineer should first understand the "
                "deployment pipeline and monitoring, then shadow production incidents.")
    ko = {
        "sections": [{"id": "kt_coverage", "title": "KT Coverage", "_coverage_rows": [{"Domain": "X", "Coverage": "Strong", "Assessment": "."}]}],
        "rendered_sections": [
            {"section_id": "first_30_day_ownership", "section_title": "30 days", "blocks": [
                {"type": "OwnershipTable", "title": "Plan", "rows": [
                    {"role": "Week 1", "team": "Understand the deployment pipeline and monitoring"},
                    {"role": "Week 2", "team": "Shadow production incidents"},
                ]},
                # Mirrors the live Groq output: the lead-in is restated too.
                {"type": "NarrativeBlock", "title": "More", "paragraphs": [
                    "During the first thirty days, the incoming engineer should follow this sequence:",
                ]},
            ]},
            {"section_id": "kt_coverage", "section_title": "KT Coverage", "blocks": []},
        ],
    }
    from knowledge.knowledge_builder import _dedup_normalize
    result = verify_document_coverage(ko, [sentence], {_dedup_normalize(sentence): "first_30_day_ownership"})
    assert result["_knowledge_coverage_summary"]["recovered"] == 0
    # Without knowing its section, the same sentence is not accepted on
    # scattered words alone.
    ko2 = {**ko, "rendered_sections": [dict(s) for s in ko["rendered_sections"]]}
    assert verify_document_coverage(ko2, [sentence])["_knowledge_coverage_summary"]["recovered"] == 1


def test_markdown_labels_from_polish_are_not_rendered_as_facts():
    section = {"id": "day1_survival_checklist", "title": "Day 1", "fields": {},
               "coverage_content": ["**Access to request:**", "Internal architecture documentation\n\n**Safe first actions (read-only):**"]}
    rendered = append_residual_content(section, {"section_id": "day1_survival_checklist", "blocks": []})
    paragraphs = [p for b in rendered["blocks"] for p in b.get("paragraphs", [])]
    assert paragraphs == ["Internal architecture documentation"]
