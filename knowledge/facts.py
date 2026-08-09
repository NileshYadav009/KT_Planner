from typing import Any, Dict, List


def build_fact(field_id: str, field: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": field_id,
        "label": field.get("label", field_id),
        "type": field.get("type", "text"),
        "value": field.get("value"),
        "confidence": float(field.get("confidence", 0.0) or 0.0),
        "source": field.get("source", "unfilled"),
        "evidence": field.get("evidence", []),
    }


def build_facts(section_id: str, fields: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [build_fact(fid, field) for fid, field in fields.items()]
