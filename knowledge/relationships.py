from typing import Any, Dict, List


def build_relationship(subject: str, relation: str, target: str) -> Dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "target": target,
    }


def build_relationships(raw_fields: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build relationship triples from a section's populated fields.

    Ownership and escalation-chain relationships, built from fields that the
    Phase 6 structured-extraction fix (see ai.wrap_structured_as_fields) now
    actually populates for the ownership_escalation section. Previously this
    checked "dependent_service"/"service_name", field ids that don't exist
    anywhere in kt_schema_new.json — always producing an empty list.
    """
    relationships: List[Dict[str, Any]] = []

    app_owner = raw_fields.get("application_ownership", {}).get("value")
    if app_owner:
        relationships.append(build_relationship("Application code", "owned by", str(app_owner)))

    infra_owner = raw_fields.get("infrastructure_ownership", {}).get("value")
    if infra_owner:
        relationships.append(build_relationship("Infrastructure", "owned by", str(infra_owner)))

    escalation_chain = raw_fields.get("escalation_chain", {}).get("value")
    if escalation_chain and isinstance(escalation_chain, str):
        steps = [s.strip() for s in escalation_chain.split("->") if s.strip()]
        for step, next_step in zip(steps, steps[1:]):
            relationships.append(build_relationship(step, "escalates to", next_step))

    return relationships
