"""Tests for knowledge/facts.py, entities.py, relationships.py — the builders
that turn populated fields into the knowledge object's facts/entities/
relationships arrays. Covers two real defects fixed this session
(REPOSITORY_AUDIT.md §9j): unfilled fields leaking into `facts` with phantom
evidence, and an escalation-chain separator mismatch between the
pattern-fallback extractor and relationships.py's parser.
"""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from knowledge.facts import build_facts
from knowledge.entities import build_entities
from knowledge.relationships import build_relationships
from knowledge.knowledge_builder import (
    build_knowledge_object,
    append_unmapped_findings_section,
    enrich_operational_calendar,
    enrich_technology_summary,
    append_tribal_knowledge_section,
    append_coverage_matrix_section,
    append_quick_reference_section,
)
from field_populator import _extract_escalation_chain
from validation import validate_knowledge_object


def _sentence(text, start=0.0, end=1.0, speaker=None, audio_confidence=0.9):
    return SimpleNamespace(text=text, start=start, end=end, speaker=speaker, audio_confidence=audio_confidence)


def test_build_facts_excludes_unfilled_fields():
    fields = {
        "system_name": {"value": "Order Service", "confidence": 0.9, "source": "pattern"},
        "kt_status": {"value": "", "confidence": 0.0, "source": "unfilled"},
        "can_deploy": {"value": None, "confidence": 0.0, "source": "unfilled"},
    }
    facts = build_facts("system_overview", fields)
    assert len(facts) == 1
    assert facts[0]["id"] == "system_name"


def test_build_facts_includes_falsy_but_meaningful_values():
    # A boolean False is a real, meaningful answer (e.g. "can_deploy: No") —
    # must not be treated the same as an unfilled placeholder.
    fields = {"can_deploy": {"value": False, "confidence": 0.9, "source": "pattern"}}
    facts = build_facts("handover_completion", fields)
    assert len(facts) == 1


def test_build_entities_excludes_empty_values():
    fields = {
        "oncall_tool": {"value": "PagerDuty", "confidence": 0.9},
        "unfilled_field": {"value": "", "confidence": 0.0},
    }
    entities = build_entities(fields)
    assert len(entities) == 1
    assert entities[0]["attributes"]["field"] == "oncall_tool"


def test_build_relationships_from_ownership_fields():
    fields = {
        "application_ownership": {"value": "Developers"},
        "infrastructure_ownership": {"value": "Platform Engineers"},
    }
    relationships = build_relationships(fields)
    assert {"subject": "Application code", "relation": "owned by", "target": "Developers"} in relationships
    assert {"subject": "Infrastructure", "relation": "owned by", "target": "Platform Engineers"} in relationships


def test_build_relationships_from_escalation_chain():
    fields = {"escalation_chain": {"value": "on-call engineer -> platform manager -> head of engineering"}}
    relationships = build_relationships(fields)
    assert len(relationships) == 2
    assert relationships[0] == {"subject": "on-call engineer", "relation": "escalates to", "target": "platform manager"}
    assert relationships[1] == {"subject": "platform manager", "relation": "escalates to", "target": "head of engineering"}


def test_escalation_chain_extractor_and_relationship_parser_agree_on_separator():
    # Regression test for the separator mismatch (audit §9j / §9n docstring):
    # _extract_escalation_chain() must produce a string that
    # build_relationships() can actually split back into steps end-to-end.
    text = (
        "Escalation path starts with an on-call engineer followed by the "
        "platform engineering manager and then the head of engineering."
    )
    chain = _extract_escalation_chain(text)
    assert chain is not None
    relationships = build_relationships({"escalation_chain": {"value": chain}})
    assert len(relationships) == 2
    assert relationships[0]["relation"] == "escalates to"


def _base_knowledge_object():
    return {
        "job_id": "job1",
        "system_name": "Test System",
        "sections": [{
            "id": "system_overview", "title": "System Overview", "status": "covered",
            "confidence": 0.9, "risk": 0.1, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {},
        }],
        "summary": {"section_count": 1, "covered_sections": 1},
    }


def test_append_unmapped_findings_section_is_noop_with_no_sentences():
    ko = _base_knowledge_object()
    result = append_unmapped_findings_section(ko, [])
    assert len(result["sections"]) == 1
    assert result["summary"]["section_count"] == 1


def test_append_unmapped_findings_section_filters_short_filler_sentences():
    ko = _base_knowledge_object()
    # "Thank you." is too short (< 4 words) to be a standalone fact — a
    # generic length filter, not specific to this wording.
    result = append_unmapped_findings_section(ko, [_sentence("Thank you.")])
    assert len(result["sections"]) == 1


def test_append_unmapped_findings_section_adds_well_shaped_section():
    ko = _base_knowledge_object()
    sentences = [
        _sentence("We use Terraform, ArgoCD and Vault for provisioning and secrets."),
        _sentence("Container images are stored in Amazon ECR."),
    ]
    result = append_unmapped_findings_section(ko, sentences)

    assert len(result["sections"]) == 2
    appended = result["sections"][-1]
    assert appended["id"] == "unmapped_findings"
    assert appended["status"] == "covered"
    assert "Terraform" in appended["coverage_content"][0]
    assert "ECR" in appended["coverage_content"][1]
    assert len(appended["evidence"]) == 2

    assert result["summary"]["section_count"] == 2
    assert result["summary"]["covered_sections"] == 2

    # Must not introduce any structural validation warnings.
    assert validate_knowledge_object(result) == []


def test_append_unmapped_findings_section_mutates_and_returns_same_object():
    ko = _base_knowledge_object()
    sentences = [_sentence("We use Terraform, ArgoCD and Vault for provisioning.")]
    result = append_unmapped_findings_section(ko, sentences)
    assert result is ko


def _section(section_id, title, **extra):
    base = {
        "id": section_id, "title": title, "status": "covered", "confidence": 0.8,
        "risk": 0.0, "facts": [], "entities": [], "evidence": [], "relationships": [],
        "fields": {}, "coverage_content": [],
    }
    base.update(extra)
    return base


def test_append_unmapped_findings_section_drops_sentences_already_mapped_elsewhere():
    # context_mapper.py's classifier can independently decide the same
    # sentence both belongs in a real section AND is "unassigned" — without
    # a cross-check that sentence renders twice (once correctly, once again
    # as a fabricated appendix "gap"). A sentence whose text is already part
    # of another section's coverage_content (even folded into a larger
    # narrative paragraph) must not survive into Unmapped Findings.
    ko = {
        "sections": [{
            "id": "system_overview", "title": "System Overview", "status": "covered",
            "confidence": 0.9, "risk": 0.1, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {},
            "coverage_content": [
                "The platform consists of React frontend applications, Python FastAPI "
                "services, PostgreSQL databases, and Redis cache layers. It processes "
                "50,000 orders per day."
            ],
        }],
        "summary": {"section_count": 1, "covered_sections": 1},
    }
    sentences = [
        # Same fact as system_overview's narrative above, just with
        # different comma placement (LLM-polish vs. raw wording) — must be
        # dropped.
        _sentence("The platform consists of React frontend applications, Python FastAPI services, PostgreSQL, databases and Redis cache layers."),
        # Genuinely never mapped anywhere — must survive.
        _sentence("We also maintain a separate internal admin tool nobody has documented."),
    ]
    result = append_unmapped_findings_section(ko, sentences)
    appended = next(s for s in result["sections"] if s["id"] == "unmapped_findings")
    assert len(appended["coverage_content"]) == 1
    assert "admin tool" in appended["coverage_content"][0]


def test_append_unmapped_findings_section_drops_sentence_with_leading_discourse_marker():
    # A raw sentence can carry a discourse marker ("Coming back to
    # architecture, ...") that a real section's classifier trims off before
    # storing the fact — the unassigned copy is then a superset (not an
    # exact/substring-after-punctuation-strip match) of the mapped one. Must
    # still be recognized as the same fact.
    ko = {
        "sections": [{
            "id": "architecture_reference", "title": "Architecture Reference", "status": "covered",
            "confidence": 0.9, "risk": 0.1, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {},
            "coverage_content": ["All services run on Amazon EKS."],
        }],
        "summary": {"section_count": 1, "covered_sections": 1},
    }
    sentences = [
        _sentence("Coming back to architecture, all services run on Amazon EKS."),
        _sentence("We also maintain a separate internal admin tool nobody has documented."),
    ]
    result = append_unmapped_findings_section(ko, sentences)
    appended = next(s for s in result["sections"] if s["id"] == "unmapped_findings")
    assert len(appended["coverage_content"]) == 1
    assert "admin tool" in appended["coverage_content"][0]


def test_enrich_operational_calendar_pulls_cost_optimization_levers():
    ko = {
        "sections": [
            _section("known_bad_days", "OPERATIONAL CALENDAR"),
            _section("cost_optimization", "COST OPTIMIZATION", _structured={
                "levers": [{"lever": "Spot instances", "detail": "Used in non-production environment"}]
            }),
        ],
        "summary": {},
    }
    result = enrich_operational_calendar(ko)
    calendar = next(s for s in result["sections"] if s["id"] == "known_bad_days")
    assert calendar["_cost_patterns"] == [{"label": "Spot instances", "value": "Used in non-production environment"}]


def test_enrich_operational_calendar_noop_when_sections_missing():
    ko = {"sections": [_section("known_bad_days", "OPERATIONAL CALENDAR")], "summary": {}}
    result = enrich_operational_calendar(ko)
    calendar = result["sections"][0]
    assert "_cost_patterns" not in calendar


def test_enrich_technology_summary_pulls_monitoring_and_security_tools():
    ko = {
        "sections": [
            _section("system_overview", "SYSTEM OVERVIEW", fields={
                "key_technologies": {"value": "Terraform, ArgoCD, Vault", "confidence": 0.9, "source": "pattern"}
            }),
            _section("monitoring_observability", "MONITORING & OBSERVABILITY", fields={
                "tools": {"value": ["Prometheus", "Grafana", "CloudWatch", "PagerDuty"], "confidence": 0.75, "source": "llm_structured"}
            }),
            _section("security_controls", "SECURITY", fields={
                "security_scan_config": {"value": "Trivy for container image scanning", "confidence": 0.75, "source": "llm_structured"}
            }),
        ],
        "summary": {},
    }
    result = enrich_technology_summary(ko)
    overview = next(s for s in result["sections"] if s["id"] == "system_overview")
    tools = [t.strip() for t in overview["fields"]["key_technologies"]["value"].split(",")]
    assert set(tools) == {"Terraform", "ArgoCD", "Vault", "Prometheus", "Grafana", "CloudWatch", "PagerDuty", "Trivy"}


def test_enrich_technology_summary_dedupes_case_insensitively():
    ko = {
        "sections": [
            _section("system_overview", "SYSTEM OVERVIEW", fields={
                "key_technologies": {"value": "Trivy", "confidence": 0.9, "source": "pattern"}
            }),
            _section("security_controls", "SECURITY", fields={
                "security_scan_config": {"value": "Trivy for container image scanning", "confidence": 0.75, "source": "llm_structured"}
            }),
        ],
        "summary": {},
    }
    result = enrich_technology_summary(ko)
    overview = next(s for s in result["sections"] if s["id"] == "system_overview")
    tools = [t.strip() for t in overview["fields"]["key_technologies"]["value"].split(",")]
    assert tools.count("Trivy") == 1


def test_enrich_technology_summary_falls_back_to_raw_coverage_content():
    # When structured extraction fails/is unavailable (fields stays empty),
    # the raw sentences are still right there in coverage_content — must not
    # lose the tool mentions just because the LLM path didn't fire.
    ko = {
        "sections": [
            _section("system_overview", "SYSTEM OVERVIEW", fields={
                "key_technologies": {"value": "Terraform", "confidence": 0.9, "source": "pattern"}
            }),
            _section("monitoring_observability", "MONITORING & OBSERVABILITY", fields={},
                      coverage_content=["We use Prometheus, Grafana and CloudWatch for monitoring."]),
        ],
        "summary": {},
    }
    result = enrich_technology_summary(ko)
    overview = next(s for s in result["sections"] if s["id"] == "system_overview")
    tools = {t.strip() for t in overview["fields"]["key_technologies"]["value"].split(",")}
    assert tools == {"Terraform", "Prometheus", "Grafana", "CloudWatch"}


def test_enrich_technology_summary_noop_without_system_overview():
    ko = {"sections": [_section("monitoring_observability", "MONITORING")], "summary": {}}
    result = enrich_technology_summary(ko)
    assert result is ko


def test_append_tribal_knowledge_section_tags_marker_phrases_by_source_section():
    ko = {"sections": [], "summary": {}}
    section_content = {
        "danger_zones": {"sentences": [{"text": "Never modify Terraform state files manually."}]},
        "monitoring_observability": {"sentences": [
            {"text": "By the way, the first thing you should check is Grafana."},
            {"text": "This sentence has no marker phrase at all."},
        ]},
    }
    result = append_tribal_knowledge_section(ko, section_content)
    tribal = next(s for s in result["sections"] if s["id"] == "tribal_knowledge")
    rows = tribal["_tribal_rows"]
    assert len(rows) == 2
    classifications = {row["Classification"] for row in rows}
    assert classifications == {"Safety-critical", "Operational shortcut"}


def test_append_tribal_knowledge_section_noop_with_no_matches():
    ko = {"sections": [], "summary": {}}
    section_content = {"system_overview": {"sentences": [{"text": "Business criticality is High."}]}}
    result = append_tribal_knowledge_section(ko, section_content)
    assert not any(s["id"] == "tribal_knowledge" for s in result["sections"])


def test_append_coverage_matrix_section_buckets_directly_from_status():
    # Regression test: bucketing used to also require confidence >= 0.6 on
    # top of status == "covered", but the pipeline's confidence signal
    # (mean block-confidence) and its status signal (semantic_coverage_score,
    # already required/optional- and density-aware) are only loosely
    # correlated — real runs showed "covered" status with confidence well
    # under 0.6, so "Strong" never fired. Bucketing must now come straight
    # from status alone.
    ko = {"sections": [], "summary": {}}
    dynamic_schema = [
        {"id": "system_overview", "title": "SYSTEM OVERVIEW"},
        {"id": "architecture_reference", "title": "ARCHITECTURE REFERENCE"},
        {"id": "cost_optimization", "title": "COST OPTIMIZATION"},
    ]
    coverage = {
        "system_overview": {"status": "covered", "confidence": 0.2, "sentence_count": 5},
        "architecture_reference": {"status": "weak", "confidence": 0.9, "sentence_count": 2},
        "cost_optimization": {"status": "missing", "confidence": 0.0, "sentence_count": 0},
    }
    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    rows = {row["Domain"]: row["Coverage"] for row in matrix["_coverage_rows"]}
    assert rows == {
        "SYSTEM OVERVIEW": "Strong",
        "ARCHITECTURE REFERENCE": "Partial",
        "COST OPTIMIZATION": "Missing",
    }


def test_append_coverage_matrix_section_reports_field_level_coverage():
    # A raw sentence count says nothing about completeness (five sentences
    # might back one field or ten) — when the section has a real fields
    # schema, the assessment should say how many of those known fields
    # actually got captured instead.
    ko = {
        "sections": [{
            "id": "system_overview", "title": "SYSTEM OVERVIEW", "status": "covered",
            "confidence": 0.8, "risk": 0.0, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "coverage_content": [],
            "fields": {
                "business_criticality": {"value": "High", "source": "llm_explicit"},
                "customer_reach": {"value": "global", "source": "llm_explicit"},
                "impact_if_down": {"value": "", "source": "unfilled"},
            },
        }],
        "summary": {},
    }
    dynamic_schema = [{
        "id": "system_overview", "title": "SYSTEM OVERVIEW",
        "fields": [
            {"id": "business_criticality", "label": "Business Criticality", "type": "single_select"},
            {"id": "customer_reach", "label": "Customer Reach", "type": "text"},
            {"id": "impact_if_down", "label": "Impact if Down", "type": "text"},
        ],
    }]
    coverage = {"system_overview": {"status": "covered", "confidence": 0.8, "sentence_count": 5}}

    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    row = matrix["_coverage_rows"][0]
    assert row["Coverage"] == "Strong"
    assert "2 of 3 known field(s) captured" in row["Assessment"]
    assert "Impact if Down" in row["Assessment"]
    assert "supporting sentence" not in row["Assessment"]


def test_append_coverage_matrix_section_collects_knowledge_gaps_separately():
    ko = {"sections": [], "summary": {}}
    dynamic_schema = [
        {"id": "system_overview", "title": "SYSTEM OVERVIEW"},
        {"id": "first_30_day_ownership", "title": "FIRST 30-DAY OWNERSHIP PLAN"},
        {"id": "handover_completion", "title": "HANDOVER COMPLETION CHECK"},
    ]
    coverage = {
        "system_overview": {"status": "covered", "confidence": 0.9, "sentence_count": 5},
        "first_30_day_ownership": {"status": "missing", "confidence": 0.0, "sentence_count": 0},
        "handover_completion": {"status": "missing", "confidence": 0.0, "sentence_count": 0},
    }
    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    assert matrix["_knowledge_gaps"] == ["FIRST 30-DAY OWNERSHIP PLAN", "HANDOVER COMPLETION CHECK"]


def test_append_quick_reference_section_skips_rows_with_no_source_data():
    ko = {
        "sections": [
            _section("ownership_escalation", "OWNERSHIP", fields={
                "escalation_chain": {"value": "on-call -> manager", "confidence": 0.7, "source": "llm_structured"}
            }),
        ],
        "summary": {},
    }
    result = append_quick_reference_section(ko)
    qr = next(s for s in result["sections"] if s["id"] == "quick_reference")
    rows = qr["_quick_reference_rows"]
    assert rows == [{"Situation": "Escalation", "Immediate reference": "on-call -> manager"}]


def test_append_quick_reference_section_noop_when_nothing_available():
    ko = {"sections": [], "summary": {}}
    result = append_quick_reference_section(ko)
    assert not any(s.get("id") == "quick_reference" for s in result["sections"])


def _minimal_schema(section_id, title, fields, tier=None):
    section = {"id": section_id, "title": title, "fields": fields}
    if tier is not None:
        section["tier"] = tier
    return section


def test_build_knowledge_object_marks_llm_inferred_values_only():
    # Regression guard for the evidence-state marker: only source == "llm"
    # (field_populator.py's free-form gap-fill, no direct grounding
    # sentence) should be flagged — pattern/semantic/llm_structured are all
    # anchored to real transcript text and must render unmodified.
    dynamic_schema = [_minimal_schema("system_overview", "SYSTEM OVERVIEW", [
        {"id": "business_criticality", "type": "text"},
        {"id": "system_name", "type": "text"},
        {"id": "key_technologies", "type": "text"},
    ])]
    populated_fields = {
        "system_overview": {
            "business_criticality": {"value": "High", "confidence": 0.75, "source": "llm"},
            "system_name": {"value": "Order Service", "confidence": 0.9, "source": "pattern"},
            "key_technologies": {"value": "Terraform", "confidence": 0.65, "source": "semantic"},
        }
    }
    coverage = {"system_overview": {"status": "covered", "confidence": 0.8, "sentence_count": 3}}

    ko = build_knowledge_object("job1", coverage, dynamic_schema, populated_fields)
    fields = ko["sections"][0]["fields"]

    assert fields["business_criticality"]["value"].startswith("High")
    assert "inferred" in fields["business_criticality"]["value"]
    assert fields["system_name"]["value"] == "Order Service"
    assert fields["key_technologies"]["value"] == "Terraform"


def test_build_knowledge_object_does_not_mark_llm_explicit_values():
    # field_populator.py tags an LLM gap-fill "llm_explicit" (not "llm")
    # when the model itself judges the fact was directly stated, just
    # paraphrased (e.g. "most business critical system" -> "High"). That
    # must render unmodified, same as pattern/semantic — only the model's
    # own "I had to guess" case ("llm") gets the inferred marker.
    dynamic_schema = [_minimal_schema("system_overview", "SYSTEM OVERVIEW", [
        {"id": "business_criticality", "type": "single_select"},
        {"id": "customer_reach", "type": "text"},
    ])]
    populated_fields = {
        "system_overview": {
            "business_criticality": {"value": "High", "confidence": 0.75, "source": "llm_explicit"},
            "customer_reach": {"value": "global", "confidence": 0.75, "source": "llm_explicit"},
        }
    }
    coverage = {"system_overview": {"status": "covered", "confidence": 0.8, "sentence_count": 3}}

    ko = build_knowledge_object("job1", coverage, dynamic_schema, populated_fields)
    fields = ko["sections"][0]["fields"]

    assert fields["business_criticality"]["value"] == "High"
    assert fields["customer_reach"]["value"] == "global"


def test_build_knowledge_object_leaves_non_string_values_unmarked():
    # A boolean/list value with source "llm" must not get a string suffix
    # appended (which would silently coerce it to a string everywhere).
    dynamic_schema = [_minimal_schema("handover_completion", "HANDOVER COMPLETION CHECK", [
        {"id": "can_deploy", "type": "boolean"},
    ])]
    populated_fields = {"handover_completion": {"can_deploy": {"value": True, "confidence": 0.7, "source": "llm"}}}
    coverage = {"handover_completion": {"status": "weak", "confidence": 0.5, "sentence_count": 1}}

    ko = build_knowledge_object("job1", coverage, dynamic_schema, populated_fields)
    assert ko["sections"][0]["fields"]["can_deploy"]["value"] is True


def test_build_knowledge_object_carries_section_tier():
    dynamic_schema = [
        _minimal_schema("system_overview", "SYSTEM OVERVIEW", [], tier=None),
        _minimal_schema("cost_optimization", "COST OPTIMIZATION", [], tier="conditional"),
    ]
    ko = build_knowledge_object("job1", {}, dynamic_schema, {})
    tiers = {s["id"]: s["tier"] for s in ko["sections"]}
    assert tiers == {"system_overview": "core", "cost_optimization": "conditional"}
