from typing import Any, Dict, List


def build_block(title: str, paragraphs: List[str]) -> Dict[str, Any]:
    return {
        "type": "NarrativeBlock",
        "title": title,
        "paragraphs": [p for p in paragraphs if p],
    }
