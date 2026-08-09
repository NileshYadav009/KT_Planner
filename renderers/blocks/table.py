from typing import Any, Dict, List


def build_block(title: str, columns: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "DecisionTable",
        "title": title,
        "columns": [col for col in columns if col],
        "rows": [row for row in rows if isinstance(row, dict)],
    }
