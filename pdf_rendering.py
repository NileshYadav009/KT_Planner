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


# Digest sections restate content that already rendered in its real section
# (or, for unmapped findings, render their own content verbatim) — they have
# no residual of their own to recover.
_RESIDUAL_EXEMPT_SECTIONS = frozenset({"unmapped_findings", "tribal_knowledge", "kt_coverage", "quick_reference"})
RESIDUAL_BLOCK_TITLE = "Additional details from the KT session"
_RESIDUAL_STOPWORDS = frozenset(
    "a an and are as at be been but by can for from had has have if in into is it its of on or "
    "that the their then there these this to was were when which while will with "
    # Relation verbs: a table row states "owner | what they own" without the
    # verb, so these must not decide whether the fact is shown.
    "own owns owned use uses used using provide provides handle handles manage manages "
    "run runs include includes".split()
)
_RESIDUAL_OVERLAP = 0.8
_MD_LABEL_RE = re.compile(r"\*\*[^*\n]{1,60}:\*\*|\n+")


def _norm_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _block_strings(value) -> list:
    """Every human-visible string in a rendered block (paragraphs, list items,
    table cells, grid values, code), recursively. Titles and the block type
    are structural, not content."""
    out = []
    if isinstance(value, str):
        if value.strip():
            out.append(value)
    elif isinstance(value, dict):
        cells = []
        for key, inner in value.items():
            if key in ("type", "title", "columns", "language"):
                continue
            inner_strings = _block_strings(inner)
            out.extend(inner_strings)
            if isinstance(inner, str):
                cells.append(inner)
        # A table/grid row splits one statement across cells ("Market
        # connectivity team" | "External market-data connectivity..."), so
        # the row as a whole must also be comparable as a single string.
        if len(cells) > 1:
            out.append(" ".join(cells))
    elif isinstance(value, (list, tuple)):
        for inner in value:
            out.extend(_block_strings(inner))
    return out


def _is_represented(text: str, rendered_norms: list) -> bool:
    norm = _norm_for_match(text)
    if not norm:
        return True
    if any(norm in r for r in rendered_norms):
        return True
    words = [_word_key(w) for w in norm.split() if w not in _RESIDUAL_STOPWORDS]
    if not words:
        return True
    # Short sentences need every content word: at 80%, "The staging deployment
    # exercise has been completed successfully." (5 words) counted as shown
    # because a longer criteria sentence happened to contain 4 of them.
    required = _RESIDUAL_OVERLAP if len(words) >= 8 else 1.0
    for r in rendered_norms:
        r_words = {_word_key(w) for w in r.split()}
        if sum(1 for w in words if w in r_words) / len(words) >= required:
            return True
    return False


# Share of a transcript sentence's content words a rendered paraphrase must
# keep (same bar as knowledge_builder's post-render paraphrase check).
_PARAPHRASE_RETENTION = 0.6


def _retention(text: str, other_norm: str) -> float:
    words = [_word_key(w) for w in _norm_for_match(text).split() if w not in _RESIDUAL_STOPWORDS]
    if not words:
        return 1.0
    other = {_word_key(w) for w in other_norm.split()}
    return sum(1 for w in words if w in other) / len(words)


def _word_key(word: str) -> str:
    # Word forms compare by their first six letters, so the polish pass's
    # "Architecturally, ..." matches the rendered "Architecture wise, ..."
    # instead of re-appearing as an unrendered residual sentence.
    return word[:6] if len(word) > 6 else word


def is_text_represented(text: str, rendered_norms: list) -> bool:
    """`text` is represented when it, or every one of its sentences, appears
    in the rendered strings. A merged two-sentence chunk is often rendered as
    two separate rows, and neither row alone contains the whole chunk."""
    if _is_represented(text, rendered_norms):
        return True
    parts = [p for p in re.split(r"(?<=[.!?])\s+", str(text or "")) if p.strip()]
    return len(parts) > 1 and all(_is_represented(p, rendered_norms) for p in parts)


def append_residual_content(section: dict, rendered: dict) -> dict:
    """Guarantee every sentence classified into a section reaches that
    section's rendered output.

    Section renderers build their blocks from a handful of named fields and
    historically only fell back to the section's sentences when NO field
    filled. So as soon as one field filled, every other sentence vanished:
    on a real KT, Security rendered 1 of 5 classified sentences (Vault/KMS,
    ECR, CI scanning and SonarQube/Trivy were gone), Ownership rendered 1 of
    13, and Handover Completion rendered none of its own sentences. The
    knowledge-coverage summary still reported zero loss because it measured
    the knowledge object, not the document.

    This compares the section's content against what its renderer actually
    produced and appends whatever is not represented as a narrative block —
    verbatim, never paraphrased, and never duplicated when a renderer already
    shows the same text (substring or near-duplicate match).
    """
    if not rendered or section.get("id") in _RESIDUAL_EXEMPT_SECTIONS:
        return rendered
    content = section.get("coverage_content") or []
    if isinstance(content, str):
        content = [content]
    try:
        from renderers.blocks.common import split_bullet_blob
        items = split_bullet_blob([c for c in content if isinstance(c, str)])
    except Exception:
        items = [c for c in content if isinstance(c, str)]
    blocks = rendered.setdefault("blocks", [])
    rendered_norms = [_norm_for_match(s) for s in _block_strings(blocks) if s.strip() != NOT_COVERED_MESSAGE]
    # Candidates are the section's transcript sentences when known. The
    # polished coverage_content can be an LLM digest with "Not specified"
    # placeholders and synthesized steps ("1. Initiate deployment via
    # Jenkins."), which the net appended to Deployment as if it had been
    # said. A transcript sentence also counts as shown when a rendered
    # polished item paraphrases it.
    source = [s for s in (section.get("_source_sentences") or []) if isinstance(s, str) and s.strip()]
    paraphrase_norms = []
    if source:
        for polished in items:
            polished = _MD_LABEL_RE.sub(" ", str(polished)).strip()
            if polished and _is_represented(polished, rendered_norms):
                paraphrase_norms.append(_norm_for_match(polished))
        items = source
    residual = []
    for item in items:
        # Polish output can carry markdown section labels ("**Access to
        # request:**") that are formatting, not facts; on their own they
        # rendered as stray paragraphs.
        item = _MD_LABEL_RE.sub(" ", str(item)).strip()
        if len(item.split()) < 3:
            continue
        if not item or item == NOT_COVERED_MESSAGE:
            continue
        if _is_represented(item, rendered_norms):
            continue
        if any(_retention(item, p) >= _PARAPHRASE_RETENTION for p in paraphrase_norms):
            continue
        # A merged chunk ("The platform team owns X. The database team owns
        # Y.") can be fully rendered sentence by sentence (two ownership rows)
        # while no single rendered string holds the whole chunk. Keep only the
        # sentences that are genuinely missing.
        parts = [p for p in re.split(r"(?<=[.!?])\s+", item) if p.strip()]
        if len(parts) > 1:
            missing = [p for p in parts if not _is_represented(p, rendered_norms)]
            if not missing:
                continue
            item = item if len(missing) == len(parts) else " ".join(missing)
        residual.append(item)
        rendered_norms.append(_norm_for_match(item))
    if not residual:
        return rendered
    # A placeholder claiming the section was not covered must not sit next
    # to content proving it was.
    rendered["blocks"] = [
        b for b in blocks
        if not (b.get("type") == "NarrativeBlock"
                and [p for p in b.get("paragraphs", []) if str(p).strip()] == [NOT_COVERED_MESSAGE])
    ]
    rendered["blocks"].append({"type": "NarrativeBlock", "title": RESIDUAL_BLOCK_TITLE, "paragraphs": residual})
    return rendered


def build_rendered_sections(knowledge_object: dict) -> list:
    rendered_sections = []
    for section in knowledge_object.get("sections", []):
        section_id = section.get("id")
        renderer = get_renderer(section_id)
        rendered = None
        if renderer:
            rendered = renderer(section)
            rendered = append_residual_content(section, rendered)

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
