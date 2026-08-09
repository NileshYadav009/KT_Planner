from typing import Any, Dict, List


def build_block(title: str, items: List[str]) -> Dict[str, Any]:
    return {
        "type": "ChecklistBlock",
        "title": title,
        "items": [item for item in items if item],
    }
