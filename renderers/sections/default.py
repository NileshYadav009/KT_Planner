from typing import Dict, Any
from renderers.base import build_default_block


def render(section: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "section_id": section.get("id"),
        "section_title": section.get("title", section.get("id")),
        "blocks": [build_default_block(section)],
    }
