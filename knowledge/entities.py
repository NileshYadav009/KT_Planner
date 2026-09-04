from typing import Any, Dict, List

# Same field-id substring dispatch convention field_populator.py's
# PATTERN_EXTRACTORS already uses, reused here so a field categorized as e.g.
# a "monitoring tool" there gets the same entity type here. Order matters —
# first match wins, so more specific substrings are checked first.
FIELD_ID_ENTITY_TYPE_HINTS = [
    (("escalation", "chain"), "escalation"),
    (("owner", "ownership"), "owner"),
    (("oncall", "on_call", "pagerduty"), "tool"),
    (("monitoring", "alert"), "monitoring"),
    (("database", "db_"), "database"),
    (("environment", "region"), "environment"),
    (("tool", "technolog", "stack", "vault", "scan"), "tool"),
]


def _infer_entity_type(field_id: str, field_label: str) -> str:
    lowered = f"{field_id} {field_label}".lower()
    for substrings, entity_type in FIELD_ID_ENTITY_TYPE_HINTS:
        if any(s in lowered for s in substrings):
            return entity_type
    return "fact"


def build_entity(name: str, entity_type: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "type": entity_type,
        "attributes": attributes,
    }


def build_entities(raw_fields: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = []
    for field_id, field in raw_fields.items():
        value = field.get("value")
        if not value:
            continue

        label = field.get("label", field_id)
        entity_type = _infer_entity_type(field_id, label)
        attrs = {"field": field_id, "confidence": field.get("confidence", 0.0)}

        if isinstance(value, list):
            # One entity per item (e.g. a multi_select tool list) instead of one
            # blob entity holding the whole list.
            for item in value:
                if item:
                    entities.append(build_entity(str(item), entity_type, attrs))
        else:
            entities.append(build_entity(str(value) if entity_type != "fact" else label, entity_type, {**attrs, "value": value}))
    return entities
