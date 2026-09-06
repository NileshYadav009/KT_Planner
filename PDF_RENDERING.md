# PDF Rendering

How a knowledge object becomes the final PDF, and the renderer-registry
pattern that makes each section's output structured instead of generic
narrative text.

## The renderer-registry pattern

`renderers/sections/__init__.py`'s `RENDERER_REGISTRY` maps each schema
section id to a function that takes that section's dict (from the knowledge
object) and returns `{"section_id", "section_title", "blocks": [...]}`.
`get_renderer(section_id)` falls back to `renderers/sections/default.py`'s
generic narrative renderer for any id without a specialized entry.

**This registry is the single most failure-prone part of the system** — this
session found and fixed the same bug shape repeatedly (`REPOSITORY_AUDIT.md`
§9l): a real, correct, dedicated renderer file exists but the registry
points at a different (wrong, or simply missing) function, so the good
renderer is silently unreachable. `validate_renderer_registry()` (called at
startup via `pipeline.load_models()`) catches the schema-id half of this —
asserting every section with a "dedicated renderer" claim actually has a
matching registry key AND that the key is a real schema section id. It does
**not** catch "the registry points at renderer A when renderer B is the
correct/intended one for this id" (both A and B exist as valid Python
functions) — that class of bug needs a human (or a future, more thorough
check) to notice.

`build_rendered_sections()` (`pdf_rendering.py`) calls the registered
renderer, and falls back to a generic, deduped narrative
(`_build_fallback_paragraphs()`, drawing from `coverage_content`, then
`facts`, then `evidence`, then `description`) whenever the renderer's output
isn't judged "meaningful" (`_has_meaningful_rendered_blocks()` — a
`NarrativeBlock` whose only content is the shared "not covered" message
doesn't count).

## Block types

Each renderer returns one or more typed blocks. `pdf_rendering.py`'s
`render_section_blocks()` and `static/index.html`'s `renderBlockHtml()` both
switch on `block["type"]` and must stay in sync (they render the same block
shapes to two different targets — PDF HTML and the UI's browser DOM):

| Type | Shape | Builder |
|---|---|---|
| `NarrativeBlock` | `{title, paragraphs: [str]}` | `renderers/blocks/narrative.py` |
| `ChecklistBlock` | `{title, items: [str]}` | `renderers/blocks/checklist.py` |
| `WarningBlock` | `{title, warnings: [str]}` | `renderers/blocks/warning.py` |
| `TechnologyGrid` | `{title, rows: [{label, value}]}` | `renderers/blocks/technology_grid.py` |
| `OwnershipTable` | `{title, rows: [{role, team}]}` | `renderers/blocks/ownership.py` |
| `DecisionTable` | `{title, columns: [str], rows: [{col: value}]}` | `renderers/blocks/table.py` |
| `DeploymentTimeline` | `{title, entries: [{label, description}]}` | `renderers/blocks/timeline.py` |

`renderers/blocks/common.py`'s `no_coverage_block(title)` is the single
shared "this section was not covered" fallback — used instead of the ~7
different ad hoc placeholder strings that used to exist per-renderer
(`REPOSITORY_AUDIT.md` §9l).

## Markdown handling — the asymmetry, and why

`SECTION_POLISH_PROMPTS` (`llm/prompts.py`) formats narrative text with
`**Label:**` markdown. That's fine for `NarrativeBlock` (full markdown via
`pdf_rendering.py`'s `_render_paragraph_text()` / the UI's
`marked.parseInline()`), but table cells, checklist items, and timeline
entries need `_render_inline_text()` instead — escape first, then convert
`**bold**` only, deliberately *not* full block-level markdown (which would
add unwanted `<p>` wrapping inside an already-tight table cell). Every block
type except `NarrativeBlock` uses this. Getting this wrong is exactly how
literal `**asterisks**` ended up visible in real exported PDFs
(`REPOSITORY_AUDIT.md` §9m) — the fix applies everywhere a data *value*
(not a static column header) gets rendered.

## Table CSS: `kv-table` vs. `grid-table`

`TechnologyGrid`/`OwnershipTable` are always exactly 2 columns (label/value);
`DecisionTable` can be N columns. They used to share one CSS rule that gave
column 1 a fixed 35% width — fine for a 2-column table, badly cramped for a
5-6 column one (`REPOSITORY_AUDIT.md` §9l). `pdf_rendering.py` now tags the
`<table>` element with `class="kv-table"` or `class="grid-table"` so
`kt_document.css` can size them differently — the 35%-first-column rule is
scoped to `.kv-table` only; `.grid-table` gets even column widths.

## Page layout

`@page` margin and `.document-body` padding were previously stacking to
~46mm of dead space per side on an A4 page (leaving only ~56% of the page
width for content) — reduced this session (§9m) to reclaim usable width,
particularly important for wide `DecisionTable`s.

## WeasyPrint specifics

`render_pdf_html()` loads `pdf/templates/kt_document.html` (Jinja2) and
`kt_document.css` (inlined into a `<style>` tag — WeasyPrint renders from a
single HTML string, no external stylesheet fetch). `api/routes.py`'s
`/export/pdf/{job_id}` calls `HTML(string=html_doc, ...).write_pdf()`
directly — no intermediate file.
