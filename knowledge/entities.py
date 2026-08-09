from typing import Any, Dict, List


def build_entity(name: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "attributes": attributes,
    }


def build_entities(raw_fields: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = []
    for field_id, field in raw_fields.items():
        value = field.get("value")
        if value:
            entities.append(build_entity(field.get("label", field_id), {"value": value}))
    return entities
