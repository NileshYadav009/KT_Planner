"""HTML/PDF document composition.

Pure functions that turn a knowledge object's rendered sections into the HTML
document string handed to WeasyPrint. No FastAPI dependency — moved out of
main.py as part of the Phase 3 architecture split (see REPOSITORY_AUDIT.md).
"""

import base64
import os
import re

from renderers import get_renderer
from renderers.blocks.common import NOT_COVERED_MESSAGE
from renderers.sections.architecture_reference import CONNECTIONS_TABLE_TITLE

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


# Cover + contents + up to 7 pages of content: no contents page.
SHORT_DOCUMENT_PAGES = 9
NOT_COVERED_SECTION_ID = "not_covered"
NOT_COVERED_SECTION_TITLE = "Not Covered in This KT"
# The combined list goes after the content sections, before these.
_DIGEST_SECTIONS = ("unmapped_findings", "tribal_knowledge", "kt_coverage", "signoff")


def _only_not_covered(section: dict) -> bool:
    blocks = section.get("blocks") or []
    return bool(blocks) and all(
        block.get("type") == "NarrativeBlock"
        and [str(p).strip() for p in block.get("paragraphs") or [] if str(p).strip()] == [NOT_COVERED_MESSAGE]
        for block in blocks)


# Recorded when the document is reviewed, not in the KT session: never a gap.
_AFTER_REVIEW_SECTION_IDS = ("signoff",)
SIGNOFF_PENDING = "Not signed yet. The sign-off is recorded when this document is reviewed and signed."


def missing_sections(knowledge_object: dict):
    """Titles of the sections the coverage matrix reports as not covered at
    all, in its order; None when the document has no coverage matrix."""
    for sec in (knowledge_object or {}).get("sections") or []:
        if sec.get("id") == "kt_coverage":
            return [str(r.get("Domain")) for r in sec.get("_coverage_rows") or []
                    if r.get("Coverage") == "Missing" and str(r.get("Assessment") or "").startswith("Not covered")]
    return None


def collapse_not_covered_sections(rendered_sections: list, missing: list = None) -> list:
    """Sections with nothing from the KT, listed once. Each used to take a
    heading and a "not covered" paragraph: six of them in a 112-word KT.

    With `missing` (missing_sections()) the list names exactly what the
    coverage matrix reports as not covered, so the list, the matrix and the
    cover agree. It used to list Sign-off (recorded at review) and leave out
    a section the document omits when empty (Handover Completion Check). A
    sign-off not yet started says so instead of being listed as a gap."""
    sections = [
        {**s, "blocks": [{"type": "NarrativeBlock", "title": s.get("section_title") or "Sign-off",
                          "paragraphs": [SIGNOFF_PENDING]}]}
        if s.get("section_id") in _AFTER_REVIEW_SECTION_IDS and _only_not_covered(s) else s
        for s in rendered_sections]
    empty = [s for s in sections if _only_not_covered(s)]
    items = [s.get("section_title") or s.get("section_id") for s in empty]
    if missing is not None:
        # A section the matrix does not call not covered keeps its place.
        listed = {str(m).lower() for m in missing}
        empty = [s for s in empty if str(s.get("section_title") or s.get("section_id")).lower() in listed]
        items = list(missing)
    if not items:
        return sections
    kept = [s for s in sections if not any(s is e for e in empty)]
    combined = {
        "section_id": NOT_COVERED_SECTION_ID,
        "section_title": NOT_COVERED_SECTION_TITLE,
        "blocks": [
            {"type": "NarrativeBlock", "title": NOT_COVERED_SECTION_TITLE,
             "paragraphs": ["Nothing in the KT session covered these sections. Confirm them with the outgoing "
                            "owner before relying on this handover."]},
            {"type": "ChecklistBlock", "title": NOT_COVERED_SECTION_TITLE, "items": items},
        ],
    }
    at = next((i for i, s in enumerate(kept) if s.get("section_id") in _DIGEST_SECTIONS), len(kept))
    return kept[:at] + [combined] + kept[at:]


def not_covered_titles(rendered_sections: list, knowledge_object: dict) -> list:
    """The section titles the PDF lists under "Not Covered in This KT"."""
    out = collapse_not_covered_sections(list(rendered_sections or []), missing_sections(knowledge_object))
    sec = next((s for s in out if s.get("section_id") == NOT_COVERED_SECTION_ID), None)
    return [str(i) for b in (sec or {}).get("blocks") or [] for i in b.get("items") or []]


def build_toc_sections(rendered_sections: list) -> list:
    toc = []
    for idx, section in enumerate(rendered_sections, start=1):
        section_title = section.get("section_title") or section.get("section_id") or f"Section {idx}"
        anchor = section.get("section_id") or f"section-{idx}"
        toc.append({"title": section_title, "anchor": anchor})
    return toc


# Link schemes a rendered paragraph may keep. Anything else (file:, data:,
# javascript:, internal hosts written as bare paths) loses its href.
_SAFE_HREF_RE = re.compile(r"^(?:https?:|mailto:|#)", re.IGNORECASE)


def _neutralise_markup(html: str) -> str:
    """Markdown output built from escaped text can still carry links and
    images written in markdown syntax. Images are dropped (nothing in a
    transcript should make the renderer fetch a resource) and links keep only
    safe schemes."""
    html = re.sub(r"<img\b[^>]*>", "", html, flags=re.IGNORECASE)

    def _href(m):
        url = m.group(2)
        return m.group(0) if _SAFE_HREF_RE.match(url) else ""

    return re.sub(r"""\s(href)\s*=\s*"([^"]*)\"""", _href, html, flags=re.IGNORECASE)


def _render_paragraph_text(text: str) -> str:
    """Paragraph text from the transcript or an LLM is untrusted: it is
    escaped before markdown conversion, so a tag in it is printed, never
    interpreted. (It used to be passed through raw whenever it contained a
    tag, which let a transcript make WeasyPrint fetch URLs and embed server
    files in the PDF.)"""
    if not isinstance(text, str):
        return ""
    content = text.strip()
    if not content:
        return ""
    escaped = html_escape(content)
    if markdown_to_html:
        return _neutralise_markup(markdown_to_html(escaped))
    return escaped


# The document's typeface, bundled so a PDF looks the same on every machine:
# the stylesheet asked for Inter, which no server had, so each fell back to
# its own font. Static, Latin subsets of Inter 4 (SIL Open Font License,
# pdf/fonts/OFL.txt), cut from the variable font for the weights the
# stylesheet uses.
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf", "fonts")
_FONT_FACES = (("Inter-Regular.ttf", 400, "normal"), ("Inter-SemiBold.ttf", 600, "normal"),
               ("Inter-Bold.ttf", 700, "normal"), ("Inter-Italic.ttf", 400, "italic"))


def _font_paths():
    return [os.path.join(FONT_DIR, name) for name, _, _ in _FONT_FACES]


def font_face_css() -> str:
    """@font-face rules for the bundled faces that are present."""
    from pathlib import Path

    rules = []
    for name, weight, style in _FONT_FACES:
        path = os.path.join(FONT_DIR, name)
        if os.path.exists(path):
            rules.append(f'@font-face {{ font-family: "Inter"; src: url("{Path(path).as_uri()}") format("truetype"); '
                         f"font-weight: {weight}; font-style: {style}; }}")
    return "\n".join(rules)


def _is_bundled_font(url: str) -> bool:
    """`url` is a file: URL of one of the bundled font files, exactly."""
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    parsed = urlparse(str(url))
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return False
    path = os.path.normcase(os.path.realpath(url2pathname(unquote(parsed.path))))
    return path in {os.path.normcase(os.path.realpath(p)) for p in _font_paths()}


def safe_url_fetcher(url, *args, **kwargs):
    """WeasyPrint URL fetcher for KT documents: only inline data: URIs are
    loaded (the stylesheet is inline and screenshots are embedded as data
    URIs), and the bundled font files by their exact paths. Any other URL in
    the document, file:// or http(s)://, is refused, so no content can make
    the renderer read server files or call internal addresses."""
    if str(url).startswith("data:") or _is_bundled_font(url):
        from weasyprint import default_url_fetcher
        return default_url_fetcher(url, *args, **kwargs)
    raise ValueError(f"External resource blocked in KT document: {str(url)[:80]}")


def _layout(html_doc: str):
    from weasyprint import HTML
    return HTML(string=html_doc, url_fetcher=safe_url_fetcher).render()


def html_to_pdf_bytes(html_doc: str) -> bytes:
    """Render a KT document with the restricted fetcher. The only entry point
    the API should use to produce a PDF."""
    return _layout(html_doc).write_pdf()


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


# An empty table cell shows a dash, and one note under the table says what it
# means: "Not covered during KT" in every empty cell filled whole rows of a
# wide table (an incident record printed it four times in one row).
NOT_COVERED_CELL = "—"
NOT_COVERED_NOTE = "Not covered during the KT; confirm with the outgoing owner."


_SVG_URI_RE = re.compile(r"data:image/svg\+xml;base64,[A-Za-z0-9+/]+={0,2}")


def _render_diagram_block(block: dict) -> str:
    """A server-drawn SVG diagram (architecture_diagram.py). Only an SVG
    data: URI is accepted, so the block cannot point the renderer at a file
    or a URL."""
    uri = block.get("svg_uri") or ""
    if not _SVG_URI_RE.fullmatch(uri):
        return ""
    return (
        "<figure class=\"diagram-figure\">"
        f"<img src=\"{uri}\" alt=\"{html_escape(block.get('title', 'Diagram'))}\"/>"
        f"<figcaption>{html_escape(block.get('caption', ''))}</figcaption></figure>"
    )


def _render_image_block(block: dict) -> str:
    """A captured screenshot, embedded in the PDF itself (data URI) so the
    exported document does not depend on the server's files."""
    path = block.get("image_path") or ""
    # Only screenshots this app captured, never an arbitrary server path.
    assets_root = os.path.realpath(os.getenv(
        "KT_ASSETS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kt_assets")))
    real = os.path.realpath(path) if path else ""
    if not real.startswith(assets_root + os.sep) or not re.fullmatch(r"screen_\d{2}\.jpg", os.path.basename(real)):
        return f"<p>{html_escape(block.get('caption', ''))} (image not available)</p>"
    try:
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return f"<p>{html_escape(block.get('caption', ''))} (image no longer available)</p>"
    parts = [
        "<figure style=\"margin:0\">",
        f"<img src=\"data:image/jpeg;base64,{data}\" style=\"width:100%;border:1px solid #d0d6e2;border-radius:4px\" alt=\"\"/>",
        f"<figcaption style=\"font-size:9pt;color:#555;margin-top:4px\">{html_escape(block.get('caption', ''))}",
    ]
    if block.get("url"):
        url = html_escape(block["url"])
        parts.append(f"<br/>Link: <a href=\"{url}\">{url}</a>")
    parts.append("</figcaption></figure>")
    return "".join(parts)


def _source_ids(block: dict, position: int) -> list:
    refs = block.get("sources") or []
    ids = refs[position] if position < len(refs) and isinstance(refs[position], list) else []
    return [int(i) for i in ids]


def _refs(block: dict, position: int) -> str:
    """Superscript links to the Sources list for one unit of a block (P1-2)."""
    ids = _source_ids(block, position)
    if not ids:
        if position in (block.get("unsourced") or []):
            return "<sup class=\"src-ref src-none\">no source</sup>"
        return ""
    links = ",".join(f"<a href=\"#src-{int(i)}\">{int(i)}</a>" for i in ids)
    return f"<sup class=\"src-ref\">{links}</sup>"


def _with_refs(paragraph_html: str, refs: str) -> str:
    """Put the reference inside the paragraph's last <p>, after its text."""
    if not refs:
        return paragraph_html
    cut = paragraph_html.rfind("</p>")
    return paragraph_html[:cut] + refs + paragraph_html[cut:] if cut >= 0 else paragraph_html + refs


def render_sources_appendix(sources: list) -> str:
    """The numbered transcript sentences the document's facts point to."""
    from knowledge.evidence import format_time

    if not sources:
        return ""
    timed = any(s.get("start") is not None for s in sources)
    attributed = any(s.get("speaker") for s in sources)
    items = []
    for src in sources:
        n = int(src["id"])
        when = format_time(src.get("start"))
        if src.get("session"):      # a KT built from several sessions (P2-1)
            when = f"S{int(src['session'])}" + (f" · {when}" if when else "")
        who = str(src.get("speaker") or "")
        items.append(f"<li id=\"src-{n}\"><span class=\"src-n\">{n}</span>"
                     + (f"<span class=\"src-t\">{when}</span>" if when else "")
                     + (f"<span class=\"src-s\">{html_escape(who)}</span>" if who else "")
                     + f"<span class=\"src-q\">\u201c{html_escape(src.get('quote', ''))}\u201d</span></li>")
    said = ("the time it was said in the recording and who said it" if timed and attributed
            else "the time it was said in the recording" if timed else "who said it" if attributed else "")
    intro = ("Each fact in this document is followed by the number of the transcript sentence it comes from"
             + (f", with {said}. " if said else ". ")
             + "Values marked inferred were not stated in the session and have no source; a fact marked "
             "“no source” could not be matched to a transcript sentence, so check it before relying on it.")
    return ("<section class=\"section-block sources-section\" id=\"sources\">"
            "<h2 class=\"section-title\"><span class=\"section-number\">&#167;</span>Sources</h2>"
            f"<p class=\"sources-intro\">{intro}</p><ol class=\"sources-list\">{''.join(items)}</ol></section>")


_COVERAGE_STATUS = {"Strong": "strong", "Partial": "partial", "Missing": "missing", "After review": "after-review"}
_MARKDOWN_STRUCTURE_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s|\*\*|#")


def _reads_as_fact_list(paragraphs: list) -> bool:
    """Three or more short, plain statements (no line breaks, lists, bold
    labels or headings of their own, as a polished section has)."""
    items = [p for p in paragraphs or [] if isinstance(p, str) and p.strip()]
    return len(items) >= 3 and len(items) == len(paragraphs) and all(
        "\n" not in p.strip() and len(p) <= 320 and not _MARKDOWN_STRUCTURE_RE.search(p) for p in items)


def handover_glance(knowledge_object: dict):
    """The cover's summary: sections covered well, partly and not at all, and
    what to confirm with the outgoing owner. None without a coverage matrix."""
    sections = knowledge_object.get("sections") or []
    coverage = next((s for s in sections if s.get("id") == "kt_coverage"), None) or {}
    rows = coverage.get("_coverage_rows") or []
    if not rows:
        return None
    counts = {"Strong": 0, "Partial": 0, "Missing": 0}
    for row in rows:
        if row.get("Coverage") in counts:
            counts[row["Coverage"]] += 1
    return {"strong": counts["Strong"], "partial": counts["Partial"], "missing": counts["Missing"],
            "follow_ups": len(coverage.get("_knowledge_gaps") or []),
            "conflicts": len(knowledge_object.get("_conflicts") or [])}


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
                paragraphs = block.get("paragraphs", [])
                if _reads_as_fact_list(paragraphs):
                    # Three or more short facts read as a list, not a wall of
                    # one-line paragraphs (PDF review, 23 Oct).
                    html.append("<ul class=\"fact-list\">")
                    for i, p in enumerate(paragraphs):
                        html.append(f"<li>{_with_refs(_render_paragraph_text(p), _refs(block, i))}</li>")
                    html.append("</ul>")
                else:
                    for i, p in enumerate(paragraphs):
                        html.append(f"<div class=\"narrative-para\">{_with_refs(_render_paragraph_text(p), _refs(block, i))}</div>")
            elif block_type == "ChecklistBlock":
                html.append("<ul>")
                for i, item in enumerate(block.get("items", [])):
                    if str(item or "").strip():   # an empty item printed a lone bullet
                        html.append(f"<li>{_render_inline_text(item)}{_refs(block, i)}</li>")
                html.append("</ul>")
            elif block_type == "WarningBlock":
                for i, warning in enumerate(block.get("warnings", [])):
                    html.append(f"<p><strong>{_render_inline_text(warning)}</strong>{_refs(block, i)}</p>")
            elif block_type == "TechnologyGrid":
                html.append("<div class=\"table-wrapper\"><table class=\"kv-table\">")
                for i, row in enumerate(block.get("rows", [])):
                    html.append(
                        f"<tr><td>{_render_inline_text(row.get('label',''))}</td>"
                        f"<td>{_render_inline_text(row.get('value',''))}{_refs(block, i)}</td></tr>"
                    )
                html.append("</table></div>")
            elif block_type == "DeploymentTimeline":
                html.append("<ol>")
                for i, entry in enumerate(block.get("entries", [])):
                    html.append(
                        f"<li><strong>{_render_inline_text(entry.get('label',''))}</strong>: "
                        f"{_render_inline_text(entry.get('description',''))}{_refs(block, i)}</li>"
                    )
                html.append("</ol>")
            elif block_type == "OwnershipTable":
                html.append("<div class=\"table-wrapper\"><table class=\"kv-table\">")
                for i, row in enumerate(block.get("rows", [])):
                    html.append(
                        f"<tr><td>{_render_inline_text(row.get('role',''))}</td>"
                        f"<td>{_render_inline_text(row.get('team',''))}{_refs(block, i)}</td></tr>"
                    )
                html.append("</table></div>")
            elif block_type == "DecisionTable":
                columns = block.get("columns", [])
                # Five or more columns at full size broke words mid-word
                # ("connection s" in a six-column incident record).
                wide = " wide-table" if len(columns) >= 5 else ""
                # The connections table repeated each KT sentence in a narrow
                # column (5-8 lines a row, over two pages for a short KT):
                # the sentence number, linked to Sources, says the same.
                by_number = raw_block_title == CONNECTIONS_TABLE_TITLE
                extra = " connections-table" if by_number else ""
                html.append(f"<div class=\"table-wrapper\"><table class=\"grid-table{wide}{extra}\">")
                html.append("<thead><tr>")
                for col in columns:
                    html.append(f"<th>{html_escape(col)}</th>")
                html.append("</tr></thead><tbody>")
                has_empty_cell = False
                for i, row in enumerate(block.get("rows", [])):
                    html.append("<tr>")
                    for c, col in enumerate(columns):
                        refs = _refs(block, i) if c == len(columns) - 1 else ""
                        cell_value = row.get(col, "")
                        if by_number and c == len(columns) - 1 and _source_ids(block, i):
                            links = ", ".join(f"<a href=\"#src-{n}\">{n}</a>" for n in _source_ids(block, i))
                            html.append(f"<td class=\"sentence-ref\">sentence {links}</td>")
                        elif col == "Coverage" and str(cell_value).strip() in _COVERAGE_STATUS:
                            # Strong / Partial / Missing as a coloured badge, so
                            # the gaps stand out in the coverage matrix.
                            status = str(cell_value).strip()
                            html.append(f"<td><span class=\"status status-{_COVERAGE_STATUS[status]}\">"
                                        f"{html_escape(status)}</span>{refs}</td>")
                        elif isinstance(cell_value, str) and cell_value.strip():
                            html.append(f"<td>{_render_inline_text(cell_value)}{refs}</td>")
                        else:
                            has_empty_cell = True
                            html.append(f"<td class=\"cell-not-covered\">{NOT_COVERED_CELL}{refs}</td>")
                    html.append("</tr>")
                html.append("</tbody></table></div>")
                if has_empty_cell:
                    html.append(f"<p class=\"table-note\"><span class=\"cell-not-covered\">{NOT_COVERED_CELL}</span> "
                                f"{NOT_COVERED_NOTE}</p>")
            elif block_type == "TroubleshootingBlock":
                html.append("<ol>")
                for i, step in enumerate(block.get("steps", [])):
                    if str(step or "").strip():
                        html.append(f"<li>{_render_inline_text(step)}{_refs(block, i)}</li>")
                html.append("</ol>")
            elif block_type == "CodeBlock":
                html.append(
                    f"<pre><code>{html_escape(block.get('code', ''))}</code></pre>"
                )
            elif block_type == "DiagramBlock":
                html.append(_render_diagram_block(block))
            elif block_type == "ImageBlock":
                html.append(_render_image_block(block))
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


def render_pdf_html(title: str, job_id: str, rendered_sections: list, coverage: dict, date_str: str,
                    warnings: list = None, notices: list = None, sources: list = None,
                    show_toc: bool = True, glance: dict = None, not_covered: list = None) -> str:
    """`warnings` (stage failures, failed LLM calls) are printed on the cover so a
    degraded document never looks like a complete one; `notices` are
    informational (e.g. built without an LLM). `sources` are the transcript
    sentences the facts point to (knowledge/evidence.py), listed at the end.
    Without `show_toc` the cover lists the sections and there is no
    contents page (build_job_pdf, for short documents). `not_covered` is the
    coverage matrix's list of sections not covered (missing_sections())."""
    env = load_template_environment()
    template = env.get_template("kt_document.html")
    rendered_sections = collapse_not_covered_sections(rendered_sections, not_covered)
    toc_sections = build_toc_sections(rendered_sections)
    content_html = render_section_blocks(rendered_sections)
    if sources:
        content_html += "\n" + render_sources_appendix(sources)
        toc_sections.append({"title": "Sources", "anchor": "sources"})
    css_path = os.path.join(os.path.dirname(__file__), "pdf", "templates", "kt_document.css")
    style_css = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            style_css = css_file.read()
    style_css = font_face_css() + "\n" + style_css

    return template.render(
        title=title,
        job_id=job_id,
        date_str=date_str,
        toc_sections=toc_sections,
        show_toc=show_toc,
        glance=glance,
        section_ids={s.get("section_id") for s in rendered_sections},
        content_html=content_html,
        style_css=style_css,
        warnings=[w for w in (warnings or []) if w],
        notices=[n for n in (notices or []) if n],
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
        elif block_type in {"ChecklistBlock", "TechnologyGrid", "DeploymentTimeline", "OwnershipTable", "DecisionTable", "TroubleshootingBlock", "WarningBlock", "ImageBlock", "DiagramBlock"}:
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
    # Who is speaking is not part of the fact: "Our RTO is two hours" is
    # shown as "RTO: two hours".
    "our we us you your "
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
            if key in ("type", "title", "columns", "language", "image_path", "src", "svg_uri"):
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


# "Our RTO is two hours and our RPO is 30 minutes" renders as two field rows:
# no one string holds the sentence, but each clause is shown, and the
# sentence was printed again under "Additional details".
_CLAUSE_SPLIT_RE = re.compile(r"\s*[,;]\s*(?:and\s+|but\s+)?|\s+(?:and|but)\s+", re.IGNORECASE)


def _content_words(text: str) -> list:
    return [_word_key(w) for w in _norm_for_match(text).split() if w not in _RESIDUAL_STOPWORDS]


def _is_represented(text: str, rendered_norms: list, by_clause: bool = True) -> bool:
    norm = _norm_for_match(text)
    if not norm:
        return True
    if any(norm in r for r in rendered_norms):
        return True
    words = _content_words(text)
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
    if by_clause:
        # Every clause shown in full somewhere. A clause of one content word
        # ("Prometheus" in "Prometheus and Grafana cover monitoring") is a
        # list item, not a statement: then the sentence is judged whole.
        clauses = [c for c in _CLAUSE_SPLIT_RE.split(text) if c and c.strip()]
        if len(clauses) > 1 and all(len(_content_words(c)) >= 2 for c in clauses):
            return all(_is_represented(c, rendered_norms, by_clause=False) for c in clauses)
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
    from dialogue import without_session_framing

    residual = []
    for item in items:
        # Polish output can carry markdown section labels ("**Access to
        # request:**") that are formatting, not facts; on their own they
        # rendered as stray paragraphs.
        item = _MD_LABEL_RE.sub(" ", str(item)).strip()
        # Compared and kept the way every section shows a sentence, without
        # the session's greeting or "I'm handing over": the field showed
        # "Our AWS e-commerce platform processes ...", so the opening
        # sentence looked missing and came back with "Hi team" in front.
        item = without_session_framing(item)
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
            not_stated = (knowledge_object.get("_not_stated") or {}).get(section_id)
            if not_stated:
                # The coverage check found the section discussed but without
                # what an engineer acts on (a fix, a rollback procedure).
                rendered.setdefault("blocks", []).append({
                    "type": "NarrativeBlock", "title": "Gaps to confirm",
                    # One sentence: the document-wide dedup drops a sentence
                    # repeated in a second section.
                    "paragraphs": [f"Not covered in the KT session: {'; '.join(not_stated)} (confirm with the "
                                   "outgoing owner before relying on this section)."],
                })
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


def build_job_pdf(job_id: str, job: dict, created: float = None) -> bytes:
    """The KT document for a finished job as PDF bytes. `created` is when
    the KT was submitted (its date on the cover). Synchronous: call it from a
    worker or a thread pool, never on the event loop."""
    from datetime import datetime

    knowledge_object = job.get("knowledge_object", {}) or {}
    title = knowledge_object.get("system_name") or job.get("title") or "KT Document"
    # The date the KT was created, not the date this copy was downloaded:
    # an older KT reopened from the Past KTs list keeps its own date.
    date_str = (datetime.fromtimestamp(created) if created else datetime.now()).strftime("%d %B %Y")
    rendered_sections = knowledge_object.get("rendered_sections")

    if not isinstance(rendered_sections, list) or not rendered_sections:
        coverage = job.get("coverage", {}) or {}
        rendered_sections = []
        for sec_id, sec_info in coverage.items():
            rendered_sections.append({
                "section_id": sec_id,
                "section_title": sec_info.get("title", sec_id),
                "blocks": [{
                    "type": "NarrativeBlock",
                    "title": sec_info.get("title", sec_id),
                    # Plain text: the renderer escapes and formats it.
                    "paragraphs": [item for item in sec_info.get("content", []) if isinstance(item, str)],
                }]
            })

    # The Sign-off section is the recorded sign-off once one has started (P1-3).
    import signoff
    rendered_sections = signoff.apply_to_document(rendered_sections, job, int(job.get("document_version") or 1))

    def _html(show_toc: bool) -> str:
        return render_pdf_html(
            title=title,
            job_id=job_id,
            rendered_sections=rendered_sections,
            coverage=job.get("coverage", {}),
            date_str=date_str,
            warnings=job.get("warnings"),
            notices=job.get("notices"),
            sources=knowledge_object.get("sources"),
            show_toc=show_toc,
            glance=handover_glance(knowledge_object),
            not_covered=missing_sections(knowledge_object),
        )

    # A contents page for a few pages of content is a page of its own for
    # little: a short document lists its sections on the cover instead.
    document = _layout(_html(True))
    if len(document.pages) <= SHORT_DOCUMENT_PAGES:
        return html_to_pdf_bytes(_html(False))
    return document.write_pdf()
