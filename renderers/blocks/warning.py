from typing import Any, Dict, List


def build_block(title: str, warnings: List[str]) -> Dict[str, Any]:
    return {
        "type": "WarningBlock",
        "title": title,
        "warnings": [w for w in warnings if w],
    }
