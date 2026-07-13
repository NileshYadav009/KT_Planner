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
