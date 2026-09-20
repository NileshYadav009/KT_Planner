from typing import Any, Dict


def build_block(title: str, code: str, language: str = "text") -> Dict[str, Any]:
    return {
        "type": "CodeBlock",
        "title": title,
        "code": code,
        "language": language,
    }
