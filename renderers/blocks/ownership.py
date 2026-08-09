from typing import Any, Dict, List


def build_block(title: str, rows: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "type": "OwnershipTable",
        "title": title,
        "rows": [row for row in rows if row.get("role") or row.get("team")],
    }
