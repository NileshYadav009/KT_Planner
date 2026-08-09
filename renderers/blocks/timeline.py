from typing import Any, Dict, List


def build_block(title: str, entries: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "type": "DeploymentTimeline",
        "title": title,
        "entries": [entry for entry in entries if entry.get("label") and entry.get("description")],
    }
