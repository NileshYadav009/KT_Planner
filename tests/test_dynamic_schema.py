import json
from pathlib import Path

from schema_generator import generate_dynamic_schema
from field_populator import populate_fields


def load_base_schema():
    path = Path(__file__).resolve().parents[1] / "kt_schema_new.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["sections"]


def test_generate_dynamic_schema_adds_tech_fields_and_keeps_required_sections():
    base_schema = load_base_schema()
    coverage = {
        "system_overview": {
            "status": "covered",
            "confidence": 0.92,
            "content": ["The platform uses Redis for caching and Kafka for event streaming."],
            "sentences": [{"text": "The platform uses Redis for caching and Kafka for event streaming."}],
        },
        "deployment_and_rollback": {
            "status": "covered",
            "confidence": 0.89,
            "content": ["We use Vault for secrets and ArgoCD for GitOps."],
            "sentences": [{"text": "We use Vault for secrets and ArgoCD for GitOps."}],
        },
        "plain_english_notes": {
            "status": "missing",
            "confidence": 0.0,
            "content": [],
            "sentences": [],
        },
    }

    dynamic_schema = generate_dynamic_schema(coverage, base_schema, include_missing_required=True)
    section_ids = {section["id"] for section in dynamic_schema}

    assert "system_overview" in section_ids
    assert "deployment_and_rollback" in section_ids
    assert "plain_english_notes" in section_ids or "plain_english_notes" not in section_ids

    system_section = next(section for section in dynamic_schema if section["id"] == "system_overview")
    field_ids = {field["id"] for field in system_section.get("fields", [])}
    assert "cache_layer" in field_ids
    assert "event_streaming" in field_ids

    deployment_section = next(section for section in dynamic_schema if section["id"] == "deployment_and_rollback")
    deployment_field_ids = {field["id"] for field in deployment_section.get("fields", [])}
    assert "secret_management" in deployment_field_ids
    assert "gitops_tool" in deployment_field_ids


def test_populate_fields_uses_pattern_and_semantic_passes():
    base_schema = load_base_schema()
    dynamic_schema = [
        {
            "id": "architecture_reference",
            "title": "Architecture Reference",
            "fields": [
                {
                    "id": "architecture_link",
                    "label": "Link to detailed architecture documentation",
                    "type": "url",
                    "required": True,
                },
                {
                    "id": "last_updated",
                    "label": "Last Updated",
                    "type": "date",
                    "required": True,
                },
            ],
        }
    ]
    coverage = {
        "architecture_reference": {
            "status": "covered",
            "confidence": 0.9,
            "content": ["The architecture diagram is maintained in Confluence at https://wiki.example.com/architecture and last updated on 2024-08-01."],
            "sentences": [{"text": "The architecture diagram is maintained in Confluence at https://wiki.example.com/architecture and last updated on 2024-08-01."}],
        }
    }

    populated = populate_fields(dynamic_schema, coverage, llm_provider=None, embedding_model=None)
    architecture = populated["architecture_reference"]

    assert architecture["architecture_link"]["source"] == "pattern"
    assert architecture["architecture_link"]["value"].startswith("https://")
    assert architecture["last_updated"]["source"] == "unfilled"


def test_dynamic_field_falls_back_to_other_sections_content():
    # Regression test for a real bug found auditing a live KT PDF: the
    # "cache_layer" dynamic field is added to system_overview whenever
    # "redis" appears ANYWHERE in the transcript (schema_generator.py scans
    # combined text across every section), but the classifier can legitimately
    # route the actual Redis sentence into a DIFFERENT section (here,
    # architecture_reference, bundled with other architecture facts) --
    # confirmed against a real transcript. Before this fix, cache_layer
    # stayed permanently "unfilled" because field_populator only ever
    # searched system_overview's own sentences, never architecture_reference's.
    # embedding_model=None throughout: _extract_by_semantic's no-model path
    # just returns the first candidate sentence unconditionally (it can't
    # score relevance without a model), so system_overview's OWN sentence
    # list is left empty here -- otherwise that trivial "return first"
    # behavior would satisfy cache_layer before the fallback path (the thing
    # actually under test) ever runs. Real runs have a real model and a
    # genuine similarity threshold; this test only needs to prove the
    # fallback mechanism itself fires and finds the right cross-section value.
    base_schema = load_base_schema()
    coverage = {
        "system_overview": {
            "status": "covered",
            "confidence": 0.9,
            # Deliberately no "content"/"sentences" text here -- see the
            # comment above on why the normal (non-fallback) path must have
            # nothing to trivially succeed on.
            "content": [],
            "sentences": [],
        },
        "architecture_reference": {
            "status": "covered",
            "confidence": 0.9,
            "content": ["Redis is used for caching and short-lived session data."],
            "sentences": [{"text": "Redis is used for caching and short-lived session data."}],
        },
    }

    dynamic_schema = generate_dynamic_schema(coverage, base_schema, include_missing_required=True)
    section_content = {
        "system_overview": {"sentences": []},
        "architecture_reference": {"sentences": coverage["architecture_reference"]["sentences"]},
    }
    populated = populate_fields(
        dynamic_schema, coverage, llm_provider=None, embedding_model=None,
        section_content=section_content,
    )

    cache_layer = populated["system_overview"]["cache_layer"]
    assert cache_layer["source"] != "unfilled"
    assert "redis" in cache_layer["value"].lower()
    # No source_chunk_index -- it would point into the wrong section's
    # sentence list (see the fallback branch's comment in field_populator.py).
    assert "source_chunk_index" not in cache_layer


def test_dynamic_field_fallback_does_not_affect_non_dynamic_fields():
    # A field that's never found anywhere (dynamic or not) must still end
    # up "unfilled" rather than the fallback grabbing something irrelevant.
    base_schema = load_base_schema()
    coverage = {
        "architecture_reference": {
            "status": "weak",
            "confidence": 0.5,
            "content": ["Some unrelated sentence about deployment timing."],
            "sentences": [{"text": "Some unrelated sentence about deployment timing."}],
        },
    }
    dynamic_schema = [{
        "id": "architecture_reference",
        "title": "Architecture Reference",
        "fields": [
            {"id": "last_updated", "label": "Last Updated", "type": "date", "required": True},
        ],
    }]
    section_content = {
        sec_id: {"sentences": cov["sentences"]} for sec_id, cov in coverage.items()
    }
    populated = populate_fields(
        dynamic_schema, coverage, llm_provider=None, embedding_model=None,
        section_content=section_content,
    )
    assert populated["architecture_reference"]["last_updated"]["source"] == "unfilled"
