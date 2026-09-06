"""Tests for knowledge/facts.py, entities.py, relationships.py — the builders
that turn populated fields into the knowledge object's facts/entities/
relationships arrays. Covers two real defects fixed this session
(REPOSITORY_AUDIT.md §9j): unfilled fields leaking into `facts` with phantom
evidence, and an escalation-chain separator mismatch between the
pattern-fallback extractor and relationships.py's parser.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from knowledge.facts import build_facts
from knowledge.entities import build_entities
from knowledge.relationships import build_relationships
from field_populator import _extract_escalation_chain


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
