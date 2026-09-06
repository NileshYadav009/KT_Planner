"""Tests for validation.py — the non-fatal structural validation layer."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from validation import (
    validate_knowledge_object,
    validate_populated_fields,
    validate_pipeline_run,
)


def _valid_ko():
    return {
        "job_id": "abc123",
        "system_name": "Test System",
        "summary": {"section_count": 1, "covered_sections": 1},
        "sections": [
            {
                "id": "system_overview",
                "title": "System Overview",
                "status": "covered",
                "confidence": 0.9,
                "risk": 0.1,
                "facts": [],
                "entities": [],
                "evidence": [],
                "relationships": [],
                "fields": {},
            }
        ],
    }


def test_valid_knowledge_object_has_no_warnings():
    assert validate_knowledge_object(_valid_ko()) == []


def test_missing_top_level_key_is_flagged():
    ko = _valid_ko()
    del ko["system_name"]
    warnings = validate_knowledge_object(ko)
    assert any("system_name" in w for w in warnings)


def test_invalid_status_is_flagged():
    ko = _valid_ko()
    ko["sections"][0]["status"] = "bogus"
    warnings = validate_knowledge_object(ko)
    assert any("status" in w and "bogus" in w for w in warnings)


def test_out_of_range_confidence_is_flagged():
    ko = _valid_ko()
    ko["sections"][0]["confidence"] = 1.5
    warnings = validate_knowledge_object(ko)
    assert any("confidence" in w for w in warnings)


def test_duplicate_section_id_is_flagged():
    ko = _valid_ko()
    ko["sections"].append(dict(ko["sections"][0]))
    warnings = validate_knowledge_object(ko)
    assert any("more than once" in w for w in warnings)


def test_not_a_dict_returns_single_warning():
    warnings = validate_knowledge_object("not a dict")
    assert len(warnings) == 1
    assert "not a dict" in warnings[0]


def _schema():
    return [
        {
            "id": "system_overview",
            "fields": [
                {"id": "business_criticality", "type": "text"},
            ],
        },
        {
            "id": "deployment_and_rollback",
            "fields": [
                {"id": "repo_link", "type": "url"},
                {
                    "id": "rollback_procedure",
                    "type": "group",
                    "fields": [
                        {"id": "rollback_trigger", "type": "text"},
                    ],
                },
            ],
        },
        # No "fields" array at all — populated via ai.wrap_structured_as_fields()'s
        # LLM structured-JSON extraction instead of field_populator.py. Real
        # schema sections (monitoring_observability, disaster_recovery, etc.)
        # are exactly this shape.
        {"id": "disaster_recovery"},
        # Same underlying case, but with a single *dynamic* bonus field
        # (schema_generator.py's TECH_STACK_FIELD_ADDITIONS) — real
        # ownership_escalation shape when the transcript mentions a paging tool.
        {"id": "ownership_escalation", "fields": [{"id": "oncall_tool", "dynamic": True}]},
    ]


def test_populated_fields_with_unknown_section_is_flagged():
    populated = {"nonexistent_section": {"foo": {"value": "x", "confidence": 0.5, "source": "pattern"}}}
    warnings = validate_populated_fields(populated, _schema())
    assert any("nonexistent_section" in w and "does not exist" in w for w in warnings)


def test_populated_fields_with_unknown_field_id_is_flagged():
    populated = {"system_overview": {"typo_field": {"value": "x", "confidence": 0.5, "source": "pattern"}}}
    warnings = validate_populated_fields(populated, _schema())
    assert any("typo_field" in w and "not a field declared" in w for w in warnings)


def test_populated_fields_skips_id_validation_for_sections_with_no_fields_array():
    # Regression test: disaster_recovery/monitoring_observability/etc. have no
    # "fields" array in the real schema (populated via LLM structured
    # extraction instead — see ai.wrap_structured_as_fields()). This used to
    # be flagged as "not a field declared" on every single real KT run,
    # dragging quality_score's validation penalty down for a correctly
    # functioning document.
    populated = {
        "disaster_recovery": {
            "rto_steps": {"value": "Restore from snapshot", "confidence": 0.75, "source": "llm_structured"},
            "recovery_contact": {"value": "Platform team", "confidence": 0.75, "source": "llm_structured"},
        }
    }
    assert validate_populated_fields(populated, _schema()) == []


def test_populated_fields_skips_id_validation_when_only_dynamic_fields_declared():
    # ownership_escalation in the real schema: no base fields, but can gain a
    # single dynamic "oncall_tool" field. The other structured-extraction
    # fields (escalation_chain, application_ownership, ...) must not be
    # flagged just because *a* field is now declared.
    populated = {
        "ownership_escalation": {
            "oncall_tool": {"value": "PagerDuty", "confidence": 0.9, "source": "pattern"},
            "escalation_chain": {"value": "engineer -> manager", "confidence": 0.75, "source": "llm_structured"},
            "application_ownership": {"value": "Developers", "confidence": 0.75, "source": "llm_structured"},
        }
    }
    assert validate_populated_fields(populated, _schema()) == []


def test_populated_fields_with_valid_field_has_no_warnings():
    populated = {"system_overview": {"business_criticality": {"value": "High", "confidence": 0.9, "source": "pattern"}}}
    assert validate_populated_fields(populated, _schema()) == []


def test_populated_fields_with_nested_group_field_is_valid():
    # rollback_trigger is nested inside the "rollback_procedure" group field —
    # must be recognized as valid without being a top-level field id.
    populated = {
        "deployment_and_rollback": {
            "rollback_procedure": {
                "rollback_trigger": {"value": "Deploy failure", "confidence": 0.8, "source": "pattern"},
            },
        },
    }
    warnings = validate_populated_fields(populated, _schema())
    assert warnings == []


def test_populated_field_missing_expected_key_is_flagged():
    populated = {"system_overview": {"business_criticality": {"value": "High", "confidence": 0.9}}}
    warnings = validate_populated_fields(populated, _schema())
    assert any("source" in w for w in warnings)


def test_populated_field_confidence_out_of_range_is_flagged():
    populated = {"system_overview": {"business_criticality": {"value": "High", "confidence": 3.0, "source": "pattern"}}}
    warnings = validate_populated_fields(populated, _schema())
    assert any("confidence" in w for w in warnings)


def test_validate_pipeline_run_combines_and_dedupes():
    ko = _valid_ko()
    ko["sections"][0]["status"] = "bogus"
    populated = {"nonexistent_section": {"foo": {"value": "x", "confidence": 0.5, "source": "pattern"}}}
    warnings = validate_pipeline_run(ko, populated_fields=populated, dynamic_schema=_schema())
    assert any("bogus" in w for w in warnings)
    assert any("nonexistent_section" in w for w in warnings)
    assert len(warnings) == len(set(warnings))


def test_validate_pipeline_run_without_populated_fields_only_checks_ko():
    ko = _valid_ko()
    warnings = validate_pipeline_run(ko)
    assert warnings == []
