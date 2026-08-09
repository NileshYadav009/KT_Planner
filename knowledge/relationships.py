from typing import Any, Dict, List


def build_relationship(subject: str, relation: str, target: str) -> Dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "target": target,
    }


def build_relationships(raw_fields: Dict[str, Any]) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []
    if raw_fields.get("dependent_service", {}).get("value") and raw_fields.get("service_name", {}).get("value"):
        relationships.append(
            build_relationship(
                raw_fields["service_name"].get("value"),
                "depends on",
                raw_fields["dependent_service"].get("value"),
            )
        )
    return relationships
