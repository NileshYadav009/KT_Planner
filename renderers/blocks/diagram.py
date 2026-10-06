import base64
from typing import Any, Dict, Optional

SVG_URI_PREFIX = "data:image/svg+xml;base64,"


def build_block(title: str, svg: str, caption: Optional[str] = None) -> Dict[str, Any]:
    """A server-drawn SVG (architecture_diagram.render_architecture_svg).
    Carried as a base64 data: URI and shown with <img>, so the PDF fetcher
    (data: URIs only) can load it and the browser never runs it as markup."""
    return {
        "type": "DiagramBlock",
        "title": title,
        "svg_uri": SVG_URI_PREFIX + base64.b64encode(svg.encode("utf-8")).decode("ascii"),
        "caption": caption or "",
    }
