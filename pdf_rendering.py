"""HTML/PDF document composition.

Pure functions that turn a knowledge object's rendered sections into the HTML
document string handed to WeasyPrint. No FastAPI dependency — moved out of
main.py as part of the Phase 3 architecture split (see REPOSITORY_AUDIT.md).
"""

import os
import re

from renderers import get_renderer
from renderers.blocks.common import NOT_COVERED_MESSAGE

try:
    from markdown import markdown as markdown_to_html
except ImportError:
    markdown_to_html = None

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:
    Environment = None


def html_escape(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return (
        value.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;")
    )


def build_toc_sections(rendered_sections: list) -> list:
    toc = []
    for idx, section in enumerate(rendered_sections, start=1):
        section_title = section.get("section_title") or section.get("section_id") or f"Section {idx}"
        anchor = section.get("section_id") or f"section-{idx}"
        toc.append({"title": section_title, "anchor": anchor})
    return toc


def _render_paragraph_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    content = text.strip()
    if not content:
        return ""
    if re.search(r"<[^>]+>", content):
        return text
    if markdown_to_html:
        return markdown_to_html(text)
    return html_escape(text)


def _render_inline_text(value) -> str:
    """Escape + convert **bold**/*italic* markdown only — for short data
    values inside table cells/checklist items/timeline entries, where full
    block-level markdown (_render_paragraph_text, which wraps output in <p>)
    would add unwanted paragraph spacing. Bold-handling exists because
    SECTION_POLISH_PROMPTS (llm/prompts.py) formats narrative with
    **Label:** markdown, and some of that polished text can end up inside a
    field value that lands in one of these block types instead of a
    NarrativeBlock — without this, the raw asterisks show up literally
    instead of rendering as bold. Italic-handling exists for
    knowledge_builder.py's INFERRED_MARKER (" *(inferred...)*"), so a
    genuinely-inferred field value reads as visually distinct from a
    transcript-grounded one wherever it's displayed.
    """
    escaped = html_escape(str(value) if value is not None else "")
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)


NOT_COVERED_CELL = "Not covered during KT"


def render_section_blocks(rendered_sections: list) -> str:
    html = []
    for idx, section in enumerate(rendered_sections, start=1):
        raw_section_title = section.get("section_title") or section.get("section_id") or "Section"
        section_title = html_escape(raw_section_title)
        section_id = section.get("section_id") or raw_section_title.lower().replace(" ", "-")
        html.append(f"<section class=\"section-block\" id=\"{html_escape(section_id)}\">")
        html.append(
            f"<h2 class=\"section-title\">"
            f"<span class=\"section-number\">{idx:02d}</span>{section_title}</h2>"
        )
        for block in section.get("blocks", []):
            raw_block_title = block.get("title") or block.get("type", "Block")
            block_type = block.get("type")
            card_classes = "block-card warning-card" if block_type == "WarningBlock" else "block-card"
            html.append(f"<div class=\"{card_classes}\">")
            # Skip the block's own heading when it just repeats the section
            # title (the common case for single-block sections) — avoids the
            # doubled-heading pattern (h2 "FOO" immediately followed by h3
            # "FOO") that showed up throughout the document.
            if raw_block_title.strip().lower() != raw_section_title.strip().lower():
                html.append(f"<h3 class=\"block-title\">{html_escape(raw_block_title)}</h3>")
            if block_type == "NarrativeBlock":
                for p in block.get("paragraphs", []):
                    html.append(f"<div class=\"narrative-para\">{_render_paragraph_text(p)}</div>")
            elif block_type == "ChecklistBlock":
                html.append("<ul>")
                for item in block.get("items", []):
                    html.append(f"<li>{_render_inline_text(item)}</li>")
                html.append("</ul>")
            elif block_type == "WarningBlock":
                for warning in block.get("warnings", []):
                    html.append(f"<p><strong>{_render_inline_text(warning)}</strong></p>")
            elif block_type == "TechnologyGrid":
                html.append("<div class=\"table-wrapper\"><table class=\"kv-table\">")
                for row in block.get("rows", []):
                    html.append(
                        f"<tr><td>{_render_inline_text(row.get('label',''))}</td>"
                        f"<td>{_render_inline_text(row.get('value',''))}</td></tr>"
                    )
                html.append("</table></div>")
            elif block_type == "DeploymentTimeline":
                html.append("<ol>")
                for entry in block.get("entries", []):
                    html.append(
                        f"<li><strong>{_render_inline_text(entry.get('label',''))}</strong>: {_render_inline_text(entry.get('description',''))}</li>"
                    )
                html.append("</ol>")
            elif block_type == "OwnershipTable":
                html.append("<div class=\"table-wrapper\"><table class=\"kv-table\">")
                for row in block.get("rows", []):
                    html.append(
                        f"<tr><td>{_render_inline_text(row.get('role',''))}</td>"
                        f"<td>{_render_inline_text(row.get('team',''))}</td></tr>"
                    )
                html.append("</table></div>")
            elif block_type == "DecisionTable":
                columns = block.get("columns", [])
                html.append("<div class=\"table-wrapper\"><table class=\"grid-table\">")
                html.append("<thead><tr>")
                for col in columns:
                    html.append(f"<th>{html_escape(col)}</th>")
                html.append("</tr></thead><tbody>")
                for row in block.get("rows", []):
                    html.append("<tr>")
                    for col in columns:
                        cell_value = row.get(col, "")
                        if isinstance(cell_value, str) and cell_value.strip():
                            html.append(f"<td>{_render_inline_text(cell_value)}</td>")
                        else:
                            html.append(f"<td class=\"cell-not-covered\">{NOT_COVERED_CELL}</td>")
                    html.append("</tr>")
                html.append("</tbody></table></div>")
            elif block_type == "TroubleshootingBlock":
                html.append("<ol>")
                for step in block.get("steps", []):
                    html.append(f"<li>{_render_inline_text(step)}</li>")
                html.append("</ol>")
            elif block_type == "CodeBlock":
                html.append(
                    f"<pre><code>{html_escape(block.get('code', ''))}</code></pre>"
                )
            else:
                html.append(f"<p>{html_escape(block.get('description', ''))}</p>")
            html.append("</div>")
        html.append("</section>")
    return "\n".join(html)


def load_template_environment():
    if Environment is None:
        raise RuntimeError("Jinja2 is not installed. Install with: pip install jinja2")
    template_dir = os.path.join(os.path.dirname(__file__), "pdf", "templates")
    loader = FileSystemLoader(template_dir)
    return Environment(loader=loader, autoescape=select_autoescape(["html", "xml"]))


def render_pdf_html(title: str, job_id: str, rendered_sections: list, coverage: dict, date_str: str) -> str:
    env = load_template_environment()
    template = env.get_template("kt_document.html")
    toc_sections = build_toc_sections(rendered_sections)
    content_html = render_section_blocks(rendered_sections)
    css_path = os.path.join(os.path.dirname(__file__), "pdf", "templates", "kt_document.css")
    style_css = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            style_css = css_file.read()

    return template.render(
        title=title,
        job_id=job_id,
        date_str=date_str,
        toc_sections=toc_sections,
        content_html=content_html,
        style_css=style_css,
    )


def _build_fallback_paragraphs(section: dict) -> list:
    seen = set()
    paragraphs = []

    def _add(text: str):
        if not isinstance(text, str):
            return
        normalized = re.sub(r"\s+", " ", text.strip()).lower()
        if not normalized:
            return
        keys = {normalized[:120]}
        if ": " in normalized:
            suffix = normalized.split(": ", 1)[1]
            keys.add(suffix[:120])
        for key in keys:
            if key in seen:
                return
        seen.update(keys)
        paragraphs.append(text.strip())

    coverage_content = section.get("coverage_content") or []
    if isinstance(coverage_content, str):
        coverage_content = [coverage_content]
    for item in coverage_content:
        _add(item)

    for fact in section.get("facts", []) or []:
        value = fact.get("value")
        if isinstance(value, str) and value.strip():
            label = fact.get("label") or fact.get("id") or "Fact"
            _add(f"{label}: {value.strip()}")

    for evidence in section.get("evidence", []) or []:
        _add(evidence.get("text", ""))

    if not paragraphs and section.get("description"):
        _add(str(section["description"]))

    return paragraphs or [NOT_COVERED_MESSAGE]


def _has_meaningful_rendered_blocks(rendered: dict) -> bool:
    blocks = rendered.get("blocks") or []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "NarrativeBlock":
            paragraphs = [p for p in block.get("paragraphs", []) if isinstance(p, str) and p.strip()]
            if paragraphs:
                if not all(p.strip() == NOT_COVERED_MESSAGE for p in paragraphs):
                    return True
        elif block_type in {"ChecklistBlock", "TechnologyGrid", "DeploymentTimeline", "OwnershipTable", "DecisionTable", "TroubleshootingBlock", "WarningBlock"}:
            return True
    return False


def build_rendered_sections(knowledge_object: dict) -> list:
    rendered_sections = []
    for section in knowledge_object.get("sections", []):
        section_id = section.get("id")
        renderer = get_renderer(section_id)
        rendered = None
        if renderer:
            rendered = renderer(section)

        if rendered and _has_meaningful_rendered_blocks(rendered):
            rendered_sections.append(rendered)
            continue

        fallback_paragraphs = _build_fallback_paragraphs(section)

        # Core sections (the default) always render, even fully empty, with
        # an explicit "not covered" placeholder — that's the point: a
        # mandatory handover area silently missing is worse than one
        # visibly flagged. Conditional sections (kt_schema_new.json's
        # "tier": "conditional" — template-mandated boilerplate areas like
        # First 30-Day Ownership Plan, not universal handover expectations)
        # are omitted entirely when truly empty instead of cluttering the
        # document with boilerplate for a topic the transcript never
        # touched and was never expected to.
        if section.get("tier") == "conditional" and fallback_paragraphs == [NOT_COVERED_MESSAGE]:
            continue

        rendered_sections.append({
            "section_id": section_id,
            "section_title": section.get("title") or section_id,
            "blocks": [{
                "type": "NarrativeBlock",
                "title": section.get("title") or section_id,
                "paragraphs": fallback_paragraphs,
            }],
        })
    return rendered_sections
